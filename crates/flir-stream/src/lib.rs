//! The FLIR picture's wire between the Infinity Manager and an in-sim gauge.
//!
//! The manager develops the thermal image beside the sim (infinity-viewports
//! nose camera → `flir-pipeline`) and the gauge — today the MD-10's head-up
//! display — draws it. They talk over an `infinity-bridge` CommBus⇄WebSocket
//! relay, which carries JSON:
//!
//! ```text
//! gauge   ──event  STATE_EVENT  (StateReport, ~15 Hz)──────────►  manager
//! manager ──command CHUNK_COMMAND (Chunk, acked, one per piece)──►  gauge
//! ```
//!
//! * The manager develops only while state reports keep arriving with
//!   `active: true`; a HUD that is stowed, killed or unloaded stops the
//!   pipeline by going quiet. Nothing has to say goodbye.
//! * A frame is 8-bit gray, square, `size × size`, encoded ([`Codec`]) and
//!   cut into base64 pieces small enough for the CommBus hop
//!   ([`CHUNK_CHARS`]). Every piece carries the frame header, so any piece
//!   can start a frame.
//! * The gauge reassembles ([`Reassembler`]) with hard limits, so garbage
//!   can never exhaust the WASM module.
//!
//! Both ends depend on THIS crate, so the names and shapes cannot drift.
//! No panics, no `unsafe`, no OS calls: it builds for `wasm32-wasip1`.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use serde::{Deserialize, Serialize};

/// Gauge → manager, fire-and-forget: a [`StateReport`].
pub const STATE_EVENT: &str = "flir.state";
/// Manager → gauge, acked: one [`Chunk`].
pub const CHUNK_COMMAND: &str = "flir.chunk";

/// Base64 characters per chunk — the size every manager→gauge stream uses
/// (terrain's tiles too): comfortably inside one CommBus call.
pub const CHUNK_CHARS: usize = 6 * 1024;

/// Largest frame side either end accepts.
pub const MAX_SIZE: u32 = 1024;
/// Most chunks a frame may claim (~1.5 MB of base64).
pub const MAX_CHUNKS: u32 = 256;
/// A frame `seq` this far BELOW the last completed one means the sender
/// restarted (it counts from the wall clock, so it never goes backwards on
/// its own); the receiver resets rather than ignoring it forever.
pub const RESTART_GAP: u64 = 100_000;

// ---------------------------------------------------------------------------
// State report
// ---------------------------------------------------------------------------

/// What the gauge tells the manager, every few tens of milliseconds while it
/// is up.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateReport {
    /// The gauge wants a picture (e.g. HUD lowered, lit, FLIR not inhibited).
    pub active: bool,
    /// Requested frame side, pixels; the manager clamps it to
    /// `64..=MAX_SIZE`.
    pub size: u32,
    /// 0 white-hot, 1 black-hot (`flir_types::Polarity`).
    #[serde(default)]
    pub polarity: u8,
    /// The weather and the aeroplane's place, for the thermal model. `None`
    /// fields fall back to the model's own estimates.
    #[serde(default)]
    pub env: EnvReport,
}

/// The environment as the aircraft reads it — `flir_types::EnvironmentWire`'s
/// fields in their sim units, every one optional (an unresolved sim var
/// is `None`, never an invented number). The manager converts.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EnvReport {
    /// `AMBIENT TEMPERATURE`, °C.
    pub temp_c: Option<f64>,
    /// `AMBIENT PRESSURE`, hPa.
    pub pressure_hpa: Option<f64>,
    /// `AMBIENT VISIBILITY`, metres.
    pub visibility_m: Option<f64>,
    /// `AMBIENT PRECIP STATE`, the sim's bitmask (2 none, 4 rain, 8 snow).
    pub precip_state: Option<f64>,
    /// `AMBIENT PRECIP RATE`, mm/h.
    pub precip_rate: Option<f64>,
    /// `AMBIENT IN CLOUD`, 0/1.
    pub in_cloud: Option<f64>,
    /// `AMBIENT WIND DIRECTION`, degrees true.
    pub wind_dir_deg: Option<f64>,
    /// `AMBIENT WIND VELOCITY`, knots.
    pub wind_kt: Option<f64>,
    /// `PLANE LATITUDE` / `PLANE LONGITUDE`, degrees.
    pub lat_deg: Option<f64>,
    pub lon_deg: Option<f64>,
    /// `PLANE ALTITUDE`, feet MSL.
    pub alt_ft: Option<f64>,
    /// `GROUND ALTITUDE`, metres MSL.
    pub ground_alt_m: Option<f64>,
    /// `E:ZULU TIME`, seconds since midnight.
    pub zulu_s: Option<f64>,
    /// `E:ZULU YEAR` and `E:ZULU DAY OF YEAR` (1-based).
    pub zulu_year: Option<f64>,
    pub zulu_day_of_year: Option<f64>,
}

impl EnvReport {
    /// Unix seconds of the sim's Zulu date and time, when all three parts
    /// are known.
    pub fn utc_unix_s(&self) -> Option<f64> {
        let (year, doy, secs) = (self.zulu_year?, self.zulu_day_of_year?, self.zulu_s?);
        if !(1970.0..=2400.0).contains(&year) || !(1.0..=366.0).contains(&doy) {
            return None;
        }
        if !secs.is_finite() || !(0.0..=86_400.0).contains(&secs) {
            return None;
        }
        Some(days_from_civil(year as i64, 1, 1) as f64 * 86_400.0 + (doy - 1.0) * 86_400.0 + secs)
    }
}

/// Days from 1970-01-01 to `y-m-d` (Howard Hinnant's algorithm).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

// ---------------------------------------------------------------------------
// Frames and the codec
// ---------------------------------------------------------------------------

/// An 8-bit gray picture, row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub seq: u64,
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

/// How a frame's bytes are packed before base64.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Codec {
    /// Raw gray bytes.
    #[serde(rename = "g8")]
    Gray8,
    /// Gray bytes, each row as left-neighbour differences, then deflate.
    /// A thermal picture is smooth, so this packs 4–8×.
    #[serde(rename = "g8d")]
    Gray8Delta,
}

/// Pack `data` (a `w × h` gray picture). `quant` > 1 rounds every value to a
/// multiple of it first — lossy, but a HUD background needs no more than
/// 64 levels, and it doubles the packing. `level` is deflate's 0–10.
pub fn encode(data: &[u8], w: u32, codec: Codec, quant: u8, level: u8) -> Vec<u8> {
    let q = quant.max(1);
    let quantised: Vec<u8>;
    let src = if q > 1 {
        let half = u16::from(q) / 2;
        quantised = data
            .iter()
            .map(|&v| {
                let r = (u16::from(v) + half) / u16::from(q) * u16::from(q);
                r.min(255) as u8
            })
            .collect();
        &quantised[..]
    } else {
        data
    };
    match codec {
        Codec::Gray8 => src.to_vec(),
        Codec::Gray8Delta => {
            let w = (w as usize).max(1);
            let mut delta = Vec::with_capacity(src.len());
            for row in src.chunks(w) {
                let mut prev = 0u8;
                for &v in row {
                    delta.push(v.wrapping_sub(prev));
                    prev = v;
                }
            }
            miniz_oxide::deflate::compress_to_vec(&delta, level.min(10))
        }
    }
}

/// Unpack what [`encode`] made into `w × h` gray bytes.
pub fn decode(bytes: &[u8], w: u32, h: u32, codec: Codec) -> Result<Vec<u8>, String> {
    let n = (w as usize) * (h as usize);
    let raw = match codec {
        Codec::Gray8 => bytes.to_vec(),
        Codec::Gray8Delta => {
            let delta = miniz_oxide::inflate::decompress_to_vec_with_limit(bytes, n)
                .map_err(|e| format!("inflate failed: {e:?}"))?;
            let w = (w as usize).max(1);
            let mut out = Vec::with_capacity(delta.len());
            for row in delta.chunks(w) {
                let mut prev = 0u8;
                for &d in row {
                    prev = prev.wrapping_add(d);
                    out.push(prev);
                }
            }
            out
        }
    };
    if raw.len() != n {
        return Err(format!("frame is {} bytes, expected {n}", raw.len()));
    }
    Ok(raw)
}

// ---------------------------------------------------------------------------
// Chunks
// ---------------------------------------------------------------------------

/// One piece of a frame, as the `flir.chunk` command's payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chunk {
    pub seq: u64,
    pub idx: u32,
    pub total: u32,
    pub w: u32,
    pub h: u32,
    pub codec: Codec,
    /// This piece of the frame's base64.
    pub data: String,
}

/// Encode `frame` and cut it into chunks of at most `chunk_chars` base64
/// characters.
pub fn chunks(frame: &Frame, codec: Codec, quant: u8, chunk_chars: usize) -> Vec<Chunk> {
    let packed = encode(&frame.data, frame.width, codec, quant, 4);
    let b64 = B64.encode(packed);
    let size = chunk_chars.max(4);
    let total = b64.len().div_ceil(size).max(1) as u32;
    let pieces: Vec<&str> = if b64.is_empty() {
        vec![""]
    } else {
        b64.as_bytes()
            .chunks(size)
            // base64 is ASCII, so every split is on a char boundary.
            .map(|c| std::str::from_utf8(c).unwrap_or(""))
            .collect()
    };
    pieces
        .into_iter()
        .enumerate()
        .map(|(idx, data)| Chunk {
            seq: frame.seq,
            idx: idx as u32,
            total,
            w: frame.width,
            h: frame.height,
            codec,
            data: data.to_string(),
        })
        .collect()
}

/// Collects chunks back into frames. Keeps at most one frame in progress —
/// the newest `seq` seen — and never goes backwards (an older frame's late
/// chunk is dropped), except across a sender restart ([`RESTART_GAP`]).
#[derive(Debug, Default)]
pub struct Reassembler {
    building: Option<Building>,
    last_done: Option<u64>,
}

#[derive(Debug)]
struct Building {
    seq: u64,
    total: u32,
    w: u32,
    h: u32,
    codec: Codec,
    parts: Vec<Option<String>>,
    have: u32,
}

impl Reassembler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one chunk. `None` while a frame is incomplete (or the chunk was
    /// stale), `Some(Ok(frame))` when one completes, `Some(Err(..))` for a
    /// chunk or frame that is refused.
    pub fn push(&mut self, c: Chunk) -> Option<Result<Frame, String>> {
        if c.total == 0 || c.total > MAX_CHUNKS || c.idx >= c.total {
            return Some(Err(format!("chunk {}/{} refused", c.idx, c.total)));
        }
        if c.w == 0 || c.h == 0 || c.w > MAX_SIZE || c.h > MAX_SIZE {
            return Some(Err(format!("frame {}x{} refused", c.w, c.h)));
        }
        if let Some(last) = self.last_done {
            if c.seq.saturating_add(RESTART_GAP) < last {
                // The sender restarted and counts from below us: start over.
                self.last_done = None;
                self.building = None;
            } else if c.seq <= last {
                return None;
            }
        }
        let fresh = match &self.building {
            Some(b) => c.seq > b.seq,
            None => true,
        };
        if fresh {
            self.building = Some(Building {
                seq: c.seq,
                total: c.total,
                w: c.w,
                h: c.h,
                codec: c.codec,
                parts: vec![None; c.total as usize],
                have: 0,
            });
        }
        let b = self.building.as_mut()?;
        if c.seq != b.seq {
            return None; // an older frame's straggler
        }
        if c.total != b.total || c.w != b.w || c.h != b.h || c.codec != b.codec {
            self.building = None;
            return Some(Err(format!("frame {} header changed mid-frame", c.seq)));
        }
        let slot = b.parts.get_mut(c.idx as usize)?;
        if slot.is_none() {
            *slot = Some(c.data);
            b.have += 1;
        }
        if b.have < b.total {
            return None;
        }
        let b = self.building.take()?;
        self.last_done = Some(b.seq);
        let joined: String = b.parts.into_iter().map(|p| p.unwrap_or_default()).collect();
        let result = B64
            .decode(joined.as_bytes())
            .map_err(|e| format!("frame {}: bad base64: {e}", b.seq))
            .and_then(|bytes| decode(&bytes, b.w, b.h, b.codec))
            .map(|data| Frame {
                seq: b.seq,
                width: b.w,
                height: b.h,
                data,
            });
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(seq: u64, n: u32) -> Frame {
        Frame {
            seq,
            width: n,
            height: n,
            data: (0..n * n).map(|i| ((i % n) * 3 + (i / n)) as u8).collect(),
        }
    }

    fn deliver(r: &mut Reassembler, cs: Vec<Chunk>) -> Option<Result<Frame, String>> {
        let mut out = None;
        for c in cs {
            if let Some(x) = r.push(c) {
                out = Some(x);
            }
        }
        out
    }

    #[test]
    fn lossless_round_trip_both_codecs() {
        for codec in [Codec::Gray8, Codec::Gray8Delta] {
            let f = frame(7, 96);
            let mut r = Reassembler::new();
            let got = deliver(&mut r, chunks(&f, codec, 1, 512));
            assert_eq!(got, Some(Ok(f)), "{codec:?}");
        }
    }

    #[test]
    fn out_of_order_chunks_still_complete() {
        let f = frame(3, 64);
        let mut cs = chunks(&f, Codec::Gray8, 1, 300);
        assert!(cs.len() > 3);
        cs.reverse();
        let mut r = Reassembler::new();
        assert_eq!(deliver(&mut r, cs), Some(Ok(f)));
    }

    #[test]
    fn quantisation_is_bounded_and_packs_smaller() {
        // A smooth picture with sensor-like grain: what quantising is for.
        let mut seed = 12345u32;
        let data: Vec<u8> = (0..128u32 * 128)
            .map(|i| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                let smooth = 100.0 + 60.0 * ((i % 128) as f32 / 20.0).sin() + (i / 128) as f32 * 0.4;
                (smooth + (seed >> 28) as f32) as u8
            })
            .collect();
        let f = Frame { seq: 1, width: 128, height: 128, data };
        let lossless = encode(&f.data, 128, Codec::Gray8Delta, 1, 4);
        let lossy = encode(&f.data, 128, Codec::Gray8Delta, 4, 4);
        assert!(lossy.len() <= lossless.len());
        let back = decode(&lossy, 128, 128, Codec::Gray8Delta).unwrap();
        assert!(back.iter().zip(&f.data).all(|(a, b)| (i16::from(*a) - i16::from(*b)).abs() <= 2));
    }

    #[test]
    fn a_newer_frame_supersedes_and_stale_chunks_are_dropped() {
        let (a, b) = (frame(10, 64), frame(11, 64));
        let ca = chunks(&a, Codec::Gray8, 1, 300);
        let cb = chunks(&b, Codec::Gray8, 1, 300);
        let mut r = Reassembler::new();
        assert_eq!(r.push(ca[0].clone()), None);
        assert_eq!(deliver(&mut r, cb), Some(Ok(b)));
        // The rest of frame 10 now arrives late and is ignored.
        assert_eq!(deliver(&mut r, ca[1..].to_vec()), None);
    }

    #[test]
    fn a_restarted_sender_is_accepted() {
        let mut r = Reassembler::new();
        let big = frame(1_000_000, 32);
        assert!(matches!(deliver(&mut r, chunks(&big, Codec::Gray8, 1, 4096)), Some(Ok(_))));
        let small = frame(5, 32);
        assert_eq!(deliver(&mut r, chunks(&small, Codec::Gray8, 1, 4096)), Some(Ok(small)));
    }

    #[test]
    fn garbage_is_refused_without_allocating() {
        let mut r = Reassembler::new();
        let bad = Chunk { seq: 1, idx: 0, total: 1_000_000, w: 8, h: 8, codec: Codec::Gray8, data: String::new() };
        assert!(matches!(r.push(bad), Some(Err(_))));
        let huge = Chunk { seq: 1, idx: 0, total: 1, w: 50_000, h: 50_000, codec: Codec::Gray8, data: String::new() };
        assert!(matches!(r.push(huge), Some(Err(_))));
        let short = Chunk { seq: 2, idx: 0, total: 1, w: 8, h: 8, codec: Codec::Gray8, data: "AAAA".into() };
        assert!(matches!(r.push(short), Some(Err(_))));
    }

    #[test]
    fn json_shapes_are_stable() {
        let c = &chunks(&frame(9, 8), Codec::Gray8Delta, 1, 6144)[0];
        let v = serde_json::to_value(c).unwrap();
        assert_eq!(v["codec"], "g8d");
        assert_eq!(v["seq"], 9);
        let s: StateReport = serde_json::from_str(r#"{"active":true,"size":512}"#).unwrap();
        assert!(s.active && s.size == 512 && s.polarity == 0 && s.env == EnvReport::default());
    }

    #[test]
    fn zulu_date_composes_to_unix_seconds() {
        let e = EnvReport {
            zulu_year: Some(2024.0),
            zulu_day_of_year: Some(173.0), // 21 June
            zulu_s: Some(0.0),
            ..EnvReport::default()
        };
        assert_eq!(e.utc_unix_s(), Some(1_718_928_000.0));
    }
}
