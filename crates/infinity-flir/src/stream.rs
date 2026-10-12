use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use serde::{Deserialize, Serialize};

pub const CHUNK_CHARS: usize = 6 * 1024;
const MAGIC: [u8; 2] = *b"FL";
const HEADER_LEN: usize = 8;
/// The wire tag of a PACKED Gray8 frame ([`Packet::to_packed_bytes`]). It
/// never appears in a [`Packet`]'s `encoding`: [`Packet::from_bytes`] unpacks
/// it straight back to [`Encoding::Gray8`].
const PACKED_GRAY8_TAG: u8 = 2;
/// zlib level for packing: the fastest. A frame is stale in tens of
/// milliseconds, so time spent squeezing it costs more than the bytes.
const PACK_LEVEL: u8 = 1;
/// The largest per-pixel error a packed frame may be sent with. Past this
/// the steps show as contours whatever the receiver does to hide them.
pub const PACK_NEAR_MAX: u8 = 8;

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Encoding {
    #[default]
    Gray8,
    Rgba8,
}

impl Encoding {
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Encoding::Gray8 => 1,
            Encoding::Rgba8 => 4,
        }
    }

    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Encoding::Gray8),
            1 => Some(Encoding::Rgba8),
            _ => None,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Packet {
    pub width: u16,
    pub height: u16,
    pub encoding: Encoding,
    pub data: Vec<u8>,
}

impl Packet {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_LEN + self.data.len());
        out.extend_from_slice(&MAGIC);
        out.push(self.encoding as u8);
        out.push(0);
        out.extend_from_slice(&self.width.to_le_bytes());
        out.extend_from_slice(&self.height.to_le_bytes());
        out.extend_from_slice(&self.data);
        out
    }

    /// [`Self::to_bytes`], but a Gray8 frame goes out PACKED: each pixel
    /// replaced by its error against the MED predictor (LOCO-I's: the median
    /// of left, up and left + up − up-left), then zlib. A FLIR picture is
    /// smooth, so the errors are mostly near zero and deflate takes the
    /// frame to a fraction of its size — which is what lets a sharper
    /// picture through the same pipe. Any other encoding goes out raw.
    ///
    /// `near` makes it NEAR-LOSSLESS, JPEG-LS's way: the error is quantised
    /// in steps of `2·near + 1`, so no pixel arrives more than `near` levels
    /// from what was sent, and the predictor runs on the RECONSTRUCTED
    /// pixels, so the bound holds for every pixel rather than drifting down
    /// a row. `0` is exact; it is clamped to [`PACK_NEAR_MAX`].
    pub fn to_packed_bytes(&self, near: u8) -> Vec<u8> {
        let (w, h) = (self.width as usize, self.height as usize);
        if self.encoding != Encoding::Gray8 || self.data.len() != w * h {
            return self.to_bytes();
        }
        let near = near.min(PACK_NEAR_MAX);
        let step = 2 * i32::from(near) + 1;
        let n = i32::from(near);
        let mut rec = self.data.clone();
        let mut residual = vec![0u8; w * h];
        let mut i = 0;
        walk_med(&mut rec, w, |p, pred| {
            let r = if near == 0 {
                residual[i] = p.wrapping_sub(pred);
                p
            } else {
                let e = i32::from(p) - i32::from(pred);
                let q = if e >= 0 {
                    (e + n) / step
                } else {
                    -((n - e) / step)
                };
                // |q| <= (255 + near) / 3 < 128: one signed byte.
                residual[i] = (q as i8) as u8;
                (i32::from(pred) + q * step).clamp(0, 255) as u8
            };
            i += 1;
            r
        });
        let z = miniz_oxide::deflate::compress_to_vec_zlib(&residual, PACK_LEVEL);
        let mut out = Vec::with_capacity(HEADER_LEN + z.len());
        out.extend_from_slice(&MAGIC);
        out.push(PACKED_GRAY8_TAG);
        out.push(near);
        out.extend_from_slice(&self.width.to_le_bytes());
        out.extend_from_slice(&self.height.to_le_bytes());
        out.extend_from_slice(&z);
        out
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self, String> {
        if b.len() < HEADER_LEN || b[..2] != MAGIC {
            return Err("not a flir packet".into());
        }
        let width = u16::from_le_bytes([b[4], b[5]]);
        let height = u16::from_le_bytes([b[6], b[7]]);
        if b[2] == PACKED_GRAY8_TAG {
            return unpack_gray8(width, height, b[3], &b[HEADER_LEN..]);
        }
        let encoding = Encoding::from_u8(b[2]).ok_or(format!("unknown encoding {}", b[2]))?;
        let data = b[HEADER_LEN..].to_vec();
        let expected = width as usize * height as usize * encoding.bytes_per_pixel();
        if data.len() != expected {
            return Err(format!(
                "expected {expected} pixel bytes, got {}",
                data.len()
            ));
        }
        Ok(Self {
            width,
            height,
            encoding,
            data,
        })
    }

    pub fn to_rgba(&self) -> Vec<u8> {
        match self.encoding {
            Encoding::Rgba8 => self.data.clone(),
            Encoding::Gray8 => self.data.iter().flat_map(|&v| [v, v, v, 255]).collect(),
        }
    }
}

/// The MED (median edge detector) prediction from a pixel's left `a`, up
/// `b` and up-left `c` neighbours.
fn med(a: u8, b: u8, c: u8) -> u8 {
    if c >= a.max(b) {
        a.min(b)
    } else if c <= a.min(b) {
        a.max(b)
    } else {
        // c lies strictly between a and b, so a + b − c is in 0..=255.
        (u16::from(a) + u16::from(b) - u16::from(c)) as u8
    }
}

/// Walk `px` (rows of `w`) in raster order, replacing each pixel with
/// `f(pixel, prediction)`, the prediction made from the pixels ALREADY
/// replaced — outside the frame reads 0. Both ends of the packing run
/// through this one walk, so they cannot disagree about a neighbour. Row by
/// row on slices: the gauge runs it on the sim's frame thread.
fn walk_med(px: &mut [u8], w: usize, mut f: impl FnMut(u8, u8) -> u8) {
    if w == 0 {
        return;
    }
    let zeros = vec![0u8; w];
    let mut start = 0;
    while start + w <= px.len() {
        let (done, rest) = px.split_at_mut(start);
        let up = done.get(start.saturating_sub(w)..).filter(|r| r.len() == w);
        let up = up.unwrap_or(&zeros);
        let (mut a, mut c) = (0u8, 0u8);
        for (v, &b) in rest.iter_mut().take(w).zip(up) {
            *v = f(*v, med(a, b, c));
            a = *v;
            c = b;
        }
        start += w;
    }
}

/// [`Packet::to_packed_bytes`]' payload back to a Gray8 packet. The
/// inflate is capped at the frame's own size, so a malformed stream can
/// never allocate more than the header claims.
fn unpack_gray8(width: u16, height: u16, near: u8, z: &[u8]) -> Result<Packet, String> {
    if near > PACK_NEAR_MAX {
        return Err(format!("packed frame near {near} refused"));
    }
    let step = 2 * i32::from(near) + 1;
    let (w, h) = (width as usize, height as usize);
    let n = w * h;
    let mut px = miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(z, n)
        .map_err(|e| format!("packed frame does not inflate: {:?}", e.status))?;
    if px.len() != n {
        return Err(format!("expected {n} packed pixels, got {}", px.len()));
    }
    if near == 0 {
        walk_med(&mut px, w, |r, pred| r.wrapping_add(pred));
    } else {
        walk_med(&mut px, w, |r, pred| {
            (i32::from(pred) + i32::from(r as i8) * step).clamp(0, 255) as u8
        });
    }
    Ok(Packet {
        width,
        height,
        encoding: Encoding::Gray8,
        data: px,
    })
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
pub struct Chunk {
    pub seq: u64,
    pub idx: usize,
    pub total: usize,
    pub data: String,
}

pub fn chunks(seq: u64, packet: &Packet, chunk_chars: usize) -> Vec<Chunk> {
    chunk_bytes(seq, &packet.to_bytes(), chunk_chars)
}

/// [`chunks`] of [`Packet::to_packed_bytes`]`(near)`. Only a receiver built
/// with this crate's packed decoding can read them.
pub fn chunks_packed(seq: u64, packet: &Packet, chunk_chars: usize, near: u8) -> Vec<Chunk> {
    chunk_bytes(seq, &packet.to_packed_bytes(near), chunk_chars)
}

fn chunk_bytes(seq: u64, bytes: &[u8], chunk_chars: usize) -> Vec<Chunk> {
    let b64 = STANDARD.encode(bytes);
    let parts: Vec<&str> = b64
        .as_bytes()
        .chunks(chunk_chars.max(4))
        .map(|c| std::str::from_utf8(c).unwrap())
        .collect();
    let total = parts.len();
    parts
        .into_iter()
        .enumerate()
        .map(|(idx, data)| Chunk {
            seq,
            idx,
            total,
            data: data.to_owned(),
        })
        .collect()
}

#[derive(Default)]
pub struct Reassembler {
    seq: u64,
    parts: Vec<Option<String>>,
    received: usize,
}

impl Reassembler {
    pub fn push(&mut self, chunk: Chunk) -> Option<Result<Packet, String>> {
        if chunk.total == 0 || chunk.idx >= chunk.total {
            return Some(Err(format!("bad chunk {}/{}", chunk.idx, chunk.total)));
        }
        if chunk.seq < self.seq {
            return None;
        }
        if chunk.seq != self.seq || self.parts.len() != chunk.total {
            self.seq = chunk.seq;
            self.parts = vec![None; chunk.total];
            self.received = 0;
        }
        let slot = &mut self.parts[chunk.idx];
        if slot.is_none() {
            self.received += 1;
        }
        *slot = Some(chunk.data);
        if self.received < self.parts.len() {
            return None;
        }
        let b64: String = self.parts.drain(..).flatten().collect();
        self.received = 0;
        self.seq += 1;
        Some(
            STANDARD
                .decode(b64)
                .map_err(|e| e.to_string())
                .and_then(|b| Packet::from_bytes(&b)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_out_of_order() {
        let packet = Packet {
            width: 64,
            height: 48,
            encoding: Encoding::Gray8,
            data: (0..64 * 48).map(|i| i as u8).collect(),
        };
        let mut parts = chunks(7, &packet, 256);
        assert!(parts.len() > 2);
        parts.reverse();
        let mut r = Reassembler::default();
        let last = parts.pop().unwrap();
        for c in parts {
            assert!(r.push(c).is_none());
        }
        assert_eq!(r.push(last).unwrap().unwrap(), packet);
    }

    /// A smooth frame with an edge in it: what a FLIR picture mostly is.
    fn smooth(w: u16, h: u16) -> Packet {
        let data = (0..h as usize)
            .flat_map(|y| {
                (0..w as usize).map(move |x| {
                    let ramp = (x + 2 * y) / 3;
                    (if x > (w as usize) / 2 {
                        ramp + 60
                    } else {
                        ramp
                    })
                    .min(255) as u8
                })
            })
            .collect();
        Packet {
            width: w,
            height: h,
            encoding: Encoding::Gray8,
            data,
        }
    }

    #[test]
    fn a_packed_frame_round_trips_exactly_and_smaller() {
        let packet = smooth(256, 200);
        let packed = packet.to_packed_bytes(0);
        assert!(
            packed.len() * 4 < packet.to_bytes().len(),
            "{} packed of {}",
            packed.len(),
            packet.to_bytes().len()
        );
        assert_eq!(Packet::from_bytes(&packed).unwrap(), packet);

        // Every byte value, every neighbour combination the predictor meets.
        let noisy = Packet {
            data: (0..256u32 * 200)
                .map(|i| (i.wrapping_mul(2_654_435_761) >> 24) as u8)
                .collect(),
            ..packet.clone()
        };
        assert_eq!(
            Packet::from_bytes(&noisy.to_packed_bytes(0)).unwrap(),
            noisy
        );

        let mut parts = chunks_packed(9, &packet, 512, 0);
        parts.reverse();
        let mut r = Reassembler::default();
        let last = parts.pop().unwrap();
        for c in parts {
            assert!(r.push(c).is_none());
        }
        assert_eq!(r.push(last).unwrap().unwrap(), packet);
    }

    /// Near-lossless: every pixel within `near` of what was sent — the
    /// noisy frame included, where an error that drifted down a row would
    /// show. (What it SAVES is a property of real pictures, which synthetic
    /// ones misstate both ways: measured on a flown capture at 512 px, ×1.4
    /// exact, ×1.8 at 1, ×2.0 at 2, ×2.2 at 3.)
    #[test]
    fn a_near_lossless_frame_is_within_near_of_every_pixel() {
        let packet = smooth(256, 200);
        let noisy = Packet {
            data: (0..256u32 * 200)
                .map(|i| (i.wrapping_mul(2_654_435_761) >> 24) as u8)
                .collect(),
            ..packet.clone()
        };
        // What a picture is: smooth, with a few levels of texture on it.
        let textured = Packet {
            data: packet
                .data
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    v.saturating_add(((i as u32).wrapping_mul(2_654_435_761) >> 30) as u8)
                })
                .collect(),
            ..packet.clone()
        };
        for p in [&packet, &noisy, &textured] {
            for near in [1u8, 2, 4, PACK_NEAR_MAX, 200] {
                let bytes = p.to_packed_bytes(near);
                let got = Packet::from_bytes(&bytes).unwrap();
                let bound = near.min(PACK_NEAR_MAX);
                let worst = p
                    .data
                    .iter()
                    .zip(&got.data)
                    .map(|(a, b)| a.abs_diff(*b))
                    .max()
                    .unwrap();
                assert!(worst <= bound, "near {near}: off by {worst}");
            }
        }
    }

    #[test]
    fn a_packed_frame_cannot_claim_more_than_its_header() {
        let small = smooth(16, 16);
        let mut over = small.to_packed_bytes(1);
        over[3] = PACK_NEAR_MAX + 1;
        assert!(Packet::from_bytes(&over).is_err(), "an unbounded near");
        let mut lie = small.to_packed_bytes(0);
        // A header claiming 8x8 over a 16x16 stream is refused, not grown.
        lie[4..6].copy_from_slice(&8u16.to_le_bytes());
        lie[6..8].copy_from_slice(&8u16.to_le_bytes());
        assert!(Packet::from_bytes(&lie).is_err());
        let mut junk = small.to_packed_bytes(0);
        junk.truncate(HEADER_LEN + 3);
        assert!(Packet::from_bytes(&junk).is_err());
        // Rgba8 has nothing to pack and goes out raw.
        let rgba = Packet {
            width: 1,
            height: 1,
            encoding: Encoding::Rgba8,
            data: vec![1, 2, 3, 4],
        };
        assert_eq!(rgba.to_packed_bytes(2), rgba.to_bytes());
    }

    #[test]
    fn stale_frames_are_dropped() {
        let packet = Packet {
            width: 1,
            height: 1,
            encoding: Encoding::Rgba8,
            data: vec![1, 2, 3, 4],
        };
        let mut r = Reassembler::default();
        let [c] = chunks(5, &packet, CHUNK_CHARS).try_into().unwrap();
        assert!(r.push(c.clone()).unwrap().is_ok());
        assert!(r.push(c).is_none());
    }
}
