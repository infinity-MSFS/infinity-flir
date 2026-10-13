use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::horizon::{HorizonCache, HorizonSettings};
use crate::stream::{Encoding, Packet};
use crate::warp::Warp;

pub struct SourceFrame {
    pub w: usize,
    pub h: usize,
    pub rgb: Vec<[f32; 3]>,
}

pub struct FlirImage {
    pub w: usize,
    pub h: usize,
    pub rgba: Vec<[u8; 4]>,
}

impl FlirImage {
    pub fn packet(&self, encoding: Encoding) -> Packet {
        let data = match encoding {
            Encoding::Rgba8 => self.rgba.as_flattened().to_vec(),
            Encoding::Gray8 => self
                .rgba
                .iter()
                .map(|&[r, g, b, _]| r.max(g).max(b))
                .collect(),
        };
        Packet {
            width: self.w as u16,
            height: self.h as u16,
            encoding,
            data,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
pub struct Crop {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Default for Crop {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            w: 1.0,
            h: 1.0,
        }
    }
}

impl Crop {
    /// This crop, taken inside `outer` rather than inside the whole frame:
    /// a user's crop of the render viewport, composed with where that
    /// viewport sits in the captured window.
    pub fn within(&self, outer: Crop) -> Crop {
        Crop {
            x: outer.x + self.x * outer.w,
            y: outer.y + self.y * outer.h,
            w: self.w * outer.w,
            h: self.h * outer.h,
        }
    }

    pub fn pixels(&self, w: usize, h: usize) -> (usize, usize, usize, usize) {
        let span = |o: f32, s: f32, n: usize| {
            let o = o.clamp(0.0, 1.0);
            let s = s.clamp(0.0, 1.0 - o);
            let start = ((o * n as f32) as usize).min(n.saturating_sub(1));
            let len = ((s * n as f32).round() as usize).clamp(1, n - start);
            (start, len)
        };
        let (x0, cw) = span(self.x, self.w, w);
        let (y0, ch) = span(self.y, self.h, h);
        (x0, y0, cw, ch)
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Polarity {
    WhiteHot,
    BlackHot,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Palette {
    Gray,
    Green,
    Amber,
    Ironbow,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(default)]
pub struct FlirSettings {
    pub sensor_width: usize,
    pub crop: Crop,
    pub weights: [f32; 3],
    pub sky_suppression: f32,
    pub persistence: f32,
    pub detail: f32,
    pub detail_radius: usize,
    pub log_domain: bool,
    pub log_floor: f32,
    pub equalize: f32,
    pub plateau: f32,
    pub agc: bool,
    pub agc_low_pct: f32,
    pub agc_high_pct: f32,
    pub agc_speed: f32,
    pub gain: f32,
    pub level: f32,
    pub gamma: f32,
    /// Scales the finished picture, 0..1. A host that draws the picture as
    /// light over the world (a HUD) uses it as transparency.
    pub brightness: f32,
    pub polarity: Polarity,
    pub palette: Palette,
    pub noise: f32,
    pub fpn: f32,
    pub vignette: f32,
    pub show_source: bool,
    pub horizon: HorizonSettings,
    pub capture_hdr: bool,
    /// The HUD ink to take back out of the capture ([`reject_ink`]), sRGB
    /// 0..1; `None` leaves the capture as it is.
    pub ink_reject: Option<[f32; 3]>,
    /// [`reject_ink`]'s refill threshold, linear ink; `<= 0` turns it off.
    pub ink_mask: f32,
    /// Resample the capture through this instead of `crop`'s rectangle
    /// ([`downsample_warped`]). Set per frame by the host from the camera,
    /// so never read from or written to a settings file.
    #[serde(skip)]
    pub warp: Option<Warp>,
}

impl Default for FlirSettings {
    fn default() -> Self {
        Self {
            sensor_width: 640,
            crop: Crop::default(),
            weights: [0.45, 0.45, 0.10],
            sky_suppression: 1.5,
            persistence: 0.3,
            detail: 1.5,
            detail_radius: 6,
            log_domain: true,
            log_floor: 0.05,
            equalize: 0.4,
            plateau: 1.0,
            agc: true,
            agc_low_pct: 1.0,
            agc_high_pct: 99.5,
            agc_speed: 0.1,
            gain: 1.0,
            level: 0.0,
            gamma: 1.0,
            brightness: 1.0,
            polarity: Polarity::WhiteHot,
            palette: Palette::Green,
            noise: 0.03,
            fpn: 0.012,
            vignette: 0.3,
            show_source: false,
            horizon: HorizonSettings::default(),
            capture_hdr: true,
            ink_reject: None,
            ink_mask: 0.35,
            warp: None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PixelFormat {
    Rgba8,
    Bgra8,
    Rgba16F,
}

pub fn downsample_raw(
    src: &[u8],
    w: usize,
    h: usize,
    pitch: usize,
    fmt: PixelFormat,
    out_w: usize,
    crop: Crop,
) -> SourceFrame {
    let (x0, y0, w, h) = crop.pixels(w, h);
    match fmt {
        PixelFormat::Rgba8 | PixelFormat::Bgra8 => {
            let (ri, bi) = if fmt == PixelFormat::Bgra8 {
                (2, 0)
            } else {
                (0, 2)
            };
            downsample_with(
                w,
                h,
                out_w,
                |x, y| {
                    let i = (y0 + y) * pitch + (x0 + x) * 4;
                    if i + 3 < src.len() {
                        [src[i + ri] as f32, src[i + 1] as f32, src[i + bi] as f32]
                    } else {
                        [0.0; 3]
                    }
                },
                |c| c.map(|v| v / 255.0),
            )
        }
        PixelFormat::Rgba16F => {
            let half = |i: usize| half::f16::from_le_bytes([src[i], src[i + 1]]).to_f32();
            downsample_with(
                w,
                h,
                out_w,
                |x, y| {
                    let i = (y0 + y) * pitch + (x0 + x) * 8;
                    if i + 7 < src.len() {
                        [half(i), half(i + 2), half(i + 4)]
                    } else {
                        [0.0; 3]
                    }
                },
                |c| c.map(|v| v.max(0.0).powf(1.0 / 2.2)),
            )
        }
    }
}

/// Most taps per axis a warped output pixel averages.
const MAX_WARP_TAPS: usize = 4;

/// Resample `src` through `warp`: output pixel `(u, v)` averages the source
/// around `warp.map(u, v)`, which is a point in `crop` — the region of the
/// capture that is the sim's render viewport (the whole frame for a
/// borderless window). A point off the viewport or behind the eye reads
/// black, which the HUD draws as clear.
#[allow(clippy::too_many_arguments)]
pub fn downsample_warped(
    src: &[u8],
    w: usize,
    h: usize,
    pitch: usize,
    fmt: PixelFormat,
    out_w: usize,
    crop: Crop,
    warp: &Warp,
) -> SourceFrame {
    let (vx, vy, vw, vh) = crop.pixels(w, h);
    let out_w = out_w.clamp(16, 4096);
    let out_h = warp.out_h(out_w);

    // Taps from the warp's footprint: the viewport pixels its unit square
    // spans per output pixel, measured along the top and left edges.
    let span = |a: Option<(f32, f32)>, b: Option<(f32, f32)>| match (a, b) {
        (Some(a), Some(b)) => ((a.0 - b.0) * vw as f32).hypot((a.1 - b.1) * vh as f32),
        _ => 0.0,
    };
    let fx = span(warp.map(0.0, 0.0), warp.map(1.0, 0.0)) / out_w as f32;
    let fy = span(warp.map(0.0, 0.0), warp.map(0.0, 1.0)) / out_h as f32;
    let taps = |f: f32| {
        if f.is_finite() {
            (f.ceil().max(1.0) as usize).min(MAX_WARP_TAPS)
        } else {
            1
        }
    };
    let (tx_n, ty_n) = (taps(fx), taps(fy));
    let norm = 1.0 / (tx_n * ty_n) as f32;

    let read = |x: usize, y: usize| -> [f32; 3] {
        match fmt {
            PixelFormat::Rgba8 | PixelFormat::Bgra8 => {
                let (ri, bi) = if fmt == PixelFormat::Bgra8 {
                    (2, 0)
                } else {
                    (0, 2)
                };
                let i = y * pitch + x * 4;
                match src.get(i..i + 4) {
                    Some(p) => [p[ri], p[1], p[bi]].map(|c| f32::from(c) / 255.0),
                    None => [0.0; 3],
                }
            }
            PixelFormat::Rgba16F => {
                let i = y * pitch + x * 8;
                match src.get(i..i + 6) {
                    Some(p) => [0, 2, 4].map(|o| {
                        let v = half::f16::from_le_bytes([p[o], p[o + 1]]).to_f32();
                        if v.is_finite() {
                            v.max(0.0).powf(1.0 / 2.2)
                        } else {
                            0.0
                        }
                    }),
                    None => [0.0; 3],
                }
            }
        }
    };

    let mut rgb = vec![[0.0f32; 3]; out_w * out_h];
    rgb.par_chunks_mut(out_w).enumerate().for_each(|(oy, row)| {
        for (ox, px) in row.iter_mut().enumerate() {
            let mut acc = [0f32; 3];
            for ty in 0..ty_n {
                let v = (oy as f32 + (ty as f32 + 0.5) / ty_n as f32) / out_h as f32;
                for tx in 0..tx_n {
                    let u = (ox as f32 + (tx as f32 + 0.5) / tx_n as f32) / out_w as f32;
                    let Some((sx, sy)) = warp.map(u, v) else {
                        continue;
                    };
                    let (x, y) = (sx * vw as f32, sy * vh as f32);
                    if !(x >= 0.0 && y >= 0.0 && x < vw as f32 && y < vh as f32) {
                        continue;
                    }
                    let c = read(vx + x as usize, vy + y as usize);
                    acc[0] += c[0];
                    acc[1] += c[1];
                    acc[2] += c[2];
                }
            }
            *px = acc.map(|v| v * norm);
        }
    });
    SourceFrame {
        w: out_w,
        h: out_h,
        rgb,
    }
}

/// The sRGB transfer, approximated as the capture's own encode is.
const GAMMA: f32 = 2.2;
/// A captured channel at or above this is clipped, and the ink under it
/// cannot be solved for.
const CLIP_LEVEL: f32 = 0.98;
/// Ink below this cannot be told from a green-leaning scene.
const INK_MIN: f32 = 0.02;
/// Radius, output pixels, a masked stroke is filled from.
const INK_FILL_RADIUS_PX: usize = 2;
/// A masked pixel with less than this fraction of clean neighbours keeps its
/// own projected value rather than a fill from almost nothing.
const INK_FILL_MIN_SUPPORT: f32 = 0.05;

/// The HUD's own light, taken back out of a capture that looks through it.
///
/// The combiner only ever ADDS one colour, `ink` (sRGB, 0..1), so a captured
/// pixel is `scene + k·ink` in linear light. Brightness measured along the
/// direction `n` — the grey axis with its ink component projected out — has
/// no ink in it at all: `L = c·n / (1,1,1)·n` is the same with or without
/// the HUD, whatever was drawn. The ink beyond a grey of that brightness,
/// `k = c·î − L·(1,1,1)·î`, is removed, never added: a scene that is LESS
/// green than grey (a blue sky) is left alone, and a green one (grass by
/// day) is pulled toward grey — the cost of the method.
///
/// Where a stroke clipped a channel, or the ink is above `mask` (tonemapping
/// is not additive), the projection is not exact; those pixels and a
/// one-pixel border are refilled from their clean neighbours. `mask <= 0`
/// turns the refill off.
pub fn reject_ink(src: &SourceFrame, ink: [f32; 3], mask: f32) -> Vec<[f32; 3]> {
    let (w, h) = (src.w, src.h);
    let lin = |c: f32| {
        if c.is_finite() {
            c.max(0.0).powf(GAMMA)
        } else {
            0.0
        }
    };
    let enc = |c: f32| c.max(0.0).powf(1.0 / GAMMA);
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

    let ink = ink.map(lin);
    let ink_len = dot(ink, ink).sqrt();
    if !(ink_len.is_finite() && ink_len > 1e-6) {
        return src.rgb.clone();
    }
    let i_hat = ink.map(|c| c / ink_len);
    let grey = [1.0; 3];
    let gi = dot(grey, i_hat);
    let n = [0, 1, 2].map(|k| grey[k] - gi * i_hat[k]);
    let gn = dot(grey, n);
    if !(gn.is_finite() && gn > 1e-6) {
        // The ink IS grey: nothing separates it from the scene.
        return src.rgb.clone();
    }

    let solved: Vec<([f32; 3], bool)> = src
        .rgb
        .par_iter()
        .map(|&c| {
            let cl = c.map(lin);
            let l = dot(cl, n) / gn;
            let k = (dot(cl, i_hat) - l * gi).max(0.0);
            let clean = [0, 1, 2].map(|j| enc(cl[j] - k * i_hat[j]));
            let clipped = c.iter().any(|&v| v >= CLIP_LEVEL) && k > INK_MIN;
            let bad = mask > 0.0 && (clipped || k > mask);
            (clean, bad)
        })
        .collect();

    let mut clean: Vec<[f32; 3]> = solved.iter().map(|s| s.0).collect();
    if mask <= 0.0 || !solved.iter().any(|s| s.1) {
        return clean;
    }

    // Grow the mask a pixel (a stroke's antialiased edge is ink too), then
    // fill it by normalised convolution of the clean pixels around it.
    let (mut tmp, mut grown, mut support) = (Vec::new(), Vec::new(), Vec::new());
    let raw: Vec<f32> = solved.iter().map(|s| f32::from(u8::from(s.1))).collect();
    box_blur(&raw, w, h, 1, &mut tmp, &mut grown);
    let valid: Vec<f32> = grown
        .iter()
        .map(|&g| f32::from(u8::from(g <= 0.0)))
        .collect();
    box_blur(&valid, w, h, INK_FILL_RADIUS_PX, &mut tmp, &mut support);
    let mut sums = Vec::new();
    for ch in 0..3 {
        let weighted: Vec<f32> = clean.iter().zip(&valid).map(|(c, v)| c[ch] * v).collect();
        box_blur(&weighted, w, h, INK_FILL_RADIUS_PX, &mut tmp, &mut sums);
        for (i, px) in clean.iter_mut().enumerate() {
            if valid[i] == 0.0 && support[i] > INK_FILL_MIN_SUPPORT {
                px[ch] = sums[i] / support[i];
            }
        }
    }
    clean
}

/// A stroke must stand this far above its 3×3 surround, linear green, to
/// teach [`estimate_ink`] the HUD's colour.
const INK_EDGE_MIN: f32 = 0.06;
/// …and be this many times greener than it is red or blue.
const INK_EDGE_GREEN_RATIO: f32 = 2.0;
/// Fewer stroke pixels than this is no measurement.
const INK_EDGE_MIN_PX: usize = 40;
/// A measured ink further than this from the nominal one is not the HUD.
const INK_MAX_DRIFT_DEG: f32 = 30.0;
/// Two inks closer than this are one direction, not a plane.
const INK_PLANE_MIN_DEG: f32 = 2.0;

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn unit3(a: [f32; 3]) -> Option<[f32; 3]> {
    let l = dot3(a, a).sqrt();
    (l.is_finite() && l > 1e-6).then(|| a.map(|c| c / l))
}

fn lin(c: f32) -> f32 {
    if c.is_finite() {
        c.max(0.0).powf(GAMMA)
    } else {
        0.0
    }
}

/// The colour the HUD actually adds to the capture, measured from it: the
/// mean direction, in linear light, of the fine green-dominant detail —
/// strokes and the FLIR picture's own edges. Unit length; `None` when too
/// few strokes are clean (unclipped) enough to say.
///
/// Why measure at all: MSFS's collimated HUD shader and its tonemapping do
/// not put the gauge's ink on screen as drawn. Flown 13 Oct 2026, the light
/// the strokes ADDED to a sky ran red/green 0.087 and blue/green 0.33 in
/// linear light, against the ink's own 0.058 and 0.16 — bluer, and taking
/// out the nominal ink left every stroke behind as a ghost.
pub fn estimate_ink(src: &SourceFrame) -> Option<[f32; 3]> {
    let (w, h) = (src.w, src.h);
    if w < 3 || h < 3 {
        return None;
    }
    let lin_px: Vec<[f32; 3]> = src.rgb.iter().map(|c| c.map(lin)).collect();
    let mut sum = [0f64; 3];
    let mut n = 0usize;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            if src.rgb[i].iter().any(|&v| v >= CLIP_LEVEL) {
                continue;
            }
            let mut mean = [0f32; 3];
            for dy in 0..3 {
                for dx in 0..3 {
                    let c = lin_px[(y + dy - 1) * w + (x + dx - 1)];
                    for k in 0..3 {
                        mean[k] += c[k] / 9.0;
                    }
                }
            }
            let c = lin_px[i];
            let d = [c[0] - mean[0], c[1] - mean[1], c[2] - mean[2]];
            if d[1] > INK_EDGE_MIN
                && d[0] >= 0.0
                && d[2] >= 0.0
                && d[1] > INK_EDGE_GREEN_RATIO * d[0].max(d[2])
            {
                for k in 0..3 {
                    sum[k] += f64::from(d[k]);
                }
                n += 1;
            }
        }
    }
    if n < INK_EDGE_MIN_PX {
        return None;
    }
    unit3(sum.map(|v| v as f32))
}

/// Brightness with none of the HUD's light in it, per pixel, ENCODED
/// (gamma 2.2, as the capture is) — what the FLIR filters when it looks
/// through its own combiner.
///
/// The HUD adds light along `ink` (the gauge's, sRGB) and, as measured,
/// along `measured` (linear, unit — [`estimate_ink`]). Brightness is read
/// along the one direction `n` orthogonal to both, so any mix of the two
/// contributes nothing, whatever was drawn and however bright; with one
/// direction (no measurement, or the two agree) it is [`reject_ink`]'s grey
/// axis with the ink projected out. `n` is mostly RED: a sky reads darker
/// and grass darker still, which is how a thermal picture draws them anyway.
///
/// Where a stroke clipped a channel the additive model fails; those pixels
/// and a one-pixel border are refilled from clean neighbours, as in
/// [`reject_ink`], when `mask > 0`.
pub fn ink_free_luminance(
    src: &SourceFrame,
    ink: [f32; 3],
    measured: Option<[f32; 3]>,
    mask: f32,
) -> Vec<f32> {
    let (w, h) = (src.w, src.h);
    let grey = [1.0f32; 3];
    let plain = || src.rgb.iter().map(|c| dot3(*c, grey) / 3.0).collect();
    let Some(i1) = unit3(ink.map(lin)) else {
        return plain();
    };
    let cross = |a: [f32; 3], b: [f32; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let plane = measured
        .and_then(unit3)
        .filter(|m| dot3(*m, i1).clamp(-1.0, 1.0).acos().to_degrees() < INK_MAX_DRIFT_DEG)
        .filter(|m| dot3(*m, i1).clamp(-1.0, 1.0).acos().to_degrees() > INK_PLANE_MIN_DEG)
        .and_then(|m| unit3(cross(i1, m)));
    let n = match plane {
        Some(n) => {
            if dot3(n, grey) < 0.0 {
                n.map(|c| -c)
            } else {
                n
            }
        }
        None => {
            let gi = dot3(grey, i1);
            [0, 1, 2].map(|k| grey[k] - gi * i1[k])
        }
    };
    let gn = dot3(grey, n);
    if !(gn.is_finite() && gn > 1e-3) {
        return plain();
    }
    let gi1 = dot3(grey, i1);
    let solved: Vec<(f32, bool)> = src
        .rgb
        .par_iter()
        .map(|&c| {
            let cl = c.map(lin);
            let l = (dot3(cl, n) / gn).max(0.0);
            // The light beyond a grey of that brightness, along the ink.
            let k = (dot3(cl, i1) - l * gi1).max(0.0);
            let clipped = c.iter().any(|&v| v >= CLIP_LEVEL) && k > INK_MIN;
            (l.powf(1.0 / GAMMA), mask > 0.0 && (clipped || k > mask))
        })
        .collect();
    let mut out: Vec<f32> = solved.iter().map(|s| s.0).collect();
    if mask <= 0.0 || !solved.iter().any(|s| s.1) {
        return out;
    }
    let (mut tmp, mut grown, mut support, mut sums) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let raw: Vec<f32> = solved.iter().map(|s| f32::from(u8::from(s.1))).collect();
    box_blur(&raw, w, h, 1, &mut tmp, &mut grown);
    let valid: Vec<f32> = grown
        .iter()
        .map(|&g| f32::from(u8::from(g <= 0.0)))
        .collect();
    box_blur(&valid, w, h, INK_FILL_RADIUS_PX, &mut tmp, &mut support);
    let weighted: Vec<f32> = out.iter().zip(&valid).map(|(l, v)| l * v).collect();
    box_blur(&weighted, w, h, INK_FILL_RADIUS_PX, &mut tmp, &mut sums);
    for (i, px) in out.iter_mut().enumerate() {
        if valid[i] == 0.0 && support[i] > INK_FILL_MIN_SUPPORT {
            *px = sums[i] / support[i];
        }
    }
    out
}

pub fn downsample_with<F, E>(w: usize, h: usize, out_w: usize, sample: F, encode: E) -> SourceFrame
where
    F: Fn(usize, usize) -> [f32; 3] + Sync,
    E: Fn([f32; 3]) -> [f32; 3] + Sync,
{
    let out_w = out_w.clamp(16, w.max(16));
    let out_h = ((out_w as f32) * h as f32 / w as f32).round().max(1.0) as usize;
    let sx = w as f32 / out_w as f32;
    let sy = h as f32 / out_h as f32;
    let taps_x = (sx.ceil() as usize).clamp(1, 4);
    let taps_y = (sy.ceil() as usize).clamp(1, 4);
    let norm = 1.0 / (taps_x * taps_y) as f32;

    let mut rgb = vec![[0.0f32; 3]; out_w * out_h];
    rgb.par_chunks_mut(out_w).enumerate().for_each(|(oy, row)| {
        for (ox, px) in row.iter_mut().enumerate() {
            let mut acc = [0f32; 3];
            for ty in 0..taps_y {
                let y = ((oy as f32 + (ty as f32 + 0.5) / taps_y as f32) * sy) as usize;
                for tx in 0..taps_x {
                    let x = ((ox as f32 + (tx as f32 + 0.5) / taps_x as f32) * sx) as usize;
                    let c = sample(x.min(w - 1), y.min(h - 1));
                    acc[0] += c[0];
                    acc[1] += c[1];
                    acc[2] += c[2];
                }
            }
            *px = encode(acc.map(|v| v * norm));
        }
    });
    SourceFrame {
        w: out_w,
        h: out_h,
        rgb,
    }
}

#[derive(Default)]
pub struct FlirProcessor {
    prev: Vec<f32>,
    fpn_cols: Vec<f32>,
    agc_lo: f32,
    agc_hi: f32,
    agc_init: bool,
    frame: u32,
    scratch: Vec<f32>,
    blur: Vec<f32>,
    eq_lut: Vec<f32>,
    last_log: Option<bool>,
    horizon: HorizonCache,
    /// The HUD's light as measured ([`estimate_ink`]), smoothed.
    ink_est: Option<[f32; 3]>,
}

/// Frames between ink measurements: the HUD's colour changes with the
/// light, not with the frame.
const INK_ESTIMATE_EVERY: u32 = 8;
/// How far one measurement moves the smoothed ink.
const INK_ESTIMATE_BLEND: f32 = 0.3;

impl FlirProcessor {
    /// The HUD colour the ink rejection is using beyond the nominal one,
    /// linear, unit — for a host's diagnostics. `None` until measured.
    pub fn measured_ink(&self) -> Option<[f32; 3]> {
        self.ink_est
    }
}

#[inline]
fn hash(mut x: u32) -> f32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    (x as f32) / (u32::MAX as f32)
}

fn box_blur(src: &[f32], w: usize, h: usize, r: usize, tmp: &mut Vec<f32>, out: &mut Vec<f32>) {
    tmp.resize(w * h, 0.0);
    out.resize(w * h, 0.0);
    let r = r.max(1) as isize;
    let inv = 1.0 / (2 * r + 1) as f32;

    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let s = &src[y * w..(y + 1) * w];
        let at = |x: isize| s[x.clamp(0, w as isize - 1) as usize];
        let mut acc: f32 = (-r..=r).map(at).sum();
        for x in 0..w as isize {
            row[x as usize] = acc * inv;
            acc += at(x + r + 1) - at(x - r);
        }
    });

    let tmp_ref = &*tmp;
    let cols: Vec<Vec<f32>> = (0..w)
        .into_par_iter()
        .map(|x| {
            let at = |y: isize| tmp_ref[y.clamp(0, h as isize - 1) as usize * w + x];
            let mut col = vec![0.0; h];
            let mut acc: f32 = (-r..=r).map(at).sum();
            for y in 0..h as isize {
                col[y as usize] = acc * inv;
                acc += at(y + r + 1) - at(y - r);
            }
            col
        })
        .collect();
    for (x, col) in cols.into_iter().enumerate() {
        for (y, v) in col.into_iter().enumerate() {
            out[y * w + x] = v;
        }
    }
}

fn ironbow(v: f32) -> [f32; 3] {
    const STOPS: [(f32, [f32; 3]); 6] = [
        (0.00, [0.00, 0.00, 0.00]),
        (0.20, [0.13, 0.00, 0.45]),
        (0.45, [0.70, 0.05, 0.55]),
        (0.70, [0.98, 0.45, 0.05]),
        (0.90, [1.00, 0.85, 0.20]),
        (1.00, [1.00, 1.00, 0.95]),
    ];
    for pair in STOPS.windows(2) {
        let (a, ca) = pair[0];
        let (b, cb) = pair[1];
        if v <= b {
            let t = ((v - a) / (b - a)).clamp(0.0, 1.0);
            return [
                ca[0] + (cb[0] - ca[0]) * t,
                ca[1] + (cb[1] - ca[1]) * t,
                ca[2] + (cb[2] - ca[2]) * t,
            ];
        }
    }
    STOPS[5].1
}

fn to_rgba8(c: [f32; 3]) -> [u8; 4] {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0) as u8;
    [q(c[0]), q(c[1]), q(c[2]), 255]
}

fn palette_rgb(p: Palette, v: f32) -> [f32; 3] {
    match p {
        Palette::Gray => [v, v, v],
        Palette::Green => [v * v * 0.45, v, v * v * 0.35],
        Palette::Amber => [v, v * 0.68, v * v * 0.12],
        Palette::Ironbow => ironbow(v),
    }
}

impl FlirProcessor {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn process(&mut self, src: &SourceFrame, s: &FlirSettings) -> FlirImage {
        let (w, h) = (src.w, src.h);
        let n = w * h;
        self.frame = self.frame.wrapping_add(1);

        // Looking through the HUD: brightness with none of its light in it
        // (`ink_free_luminance`), which replaces the colour weighting below.
        let ink_free: Option<Vec<f32>> = s.ink_reject.map(|ink| {
            if (self.ink_est.is_none() || self.frame.is_multiple_of(INK_ESTIMATE_EVERY))
                && let Some(m) = estimate_ink(src)
            {
                self.ink_est = match self.ink_est {
                    Some(p) => unit3(
                        [0, 1, 2]
                            .map(|k| p[k] * (1.0 - INK_ESTIMATE_BLEND) + m[k] * INK_ESTIMATE_BLEND),
                    ),
                    None => Some(m),
                };
            }
            ink_free_luminance(src, ink, self.ink_est, s.ink_mask)
        });
        let rgb: &[[f32; 3]] = &src.rgb;

        if s.show_source {
            let rgba = match &ink_free {
                Some(l) => l.iter().map(|&v| to_rgba8([v; 3])).collect(),
                None => rgb.iter().map(|&c| to_rgba8(c)).collect(),
            };
            return FlirImage { w, h, rgba };
        }

        let [wr, wg, wb] = s.weights;
        let sky = s.sky_suppression;
        let log_floor = if s.log_domain {
            s.log_floor.max(1e-4)
        } else {
            0.0
        };
        let log_range = if log_floor > 0.0 {
            ((1.0 + log_floor) / log_floor).ln()
        } else {
            1.0
        };
        let hz_on = s.horizon.enabled;
        let (gv, sv) = (s.horizon.ground_vis, s.horizon.sky_vis);
        let bg: &[[f32; 3]] = if hz_on {
            self.horizon.get(&s.horizon, w, h)
        } else {
            &[]
        };
        let mut heat: Vec<f32> = rgb
            .par_iter()
            .enumerate()
            .map(|(i, &[r, g, b])| {
                let mut h = match &ink_free {
                    // Already sky-dark: see `ink_free_luminance`.
                    Some(l) => l.get(i).copied().unwrap_or(0.0),
                    None => {
                        let hue_gate = ((g - r) / b.max(1e-3) * 4.0 + 0.5).clamp(0.0, 1.0);
                        let blueness = (b - r.max(g)).max(0.0) * hue_gate;
                        wr * r + wg * g + wb * b - sky * blueness
                    }
                };
                let mut bg_t = 0.0;
                if hz_on {
                    let [t, sky_w, _] = bg[i];
                    h *= gv + (sv - gv) * sky_w;
                    bg_t = t;
                }
                if log_floor > 0.0 {
                    (h.max(0.0) + log_floor).ln() + bg_t * log_range
                } else {
                    h + bg_t
                }
            })
            .collect();

        if self.last_log != Some(s.log_domain) {
            self.last_log = Some(s.log_domain);
            self.prev.clear();
            self.eq_lut.clear();
            self.agc_init = false;
        }

        if self.prev.len() == n && s.persistence > 0.0 {
            let p = s.persistence.clamp(0.0, 0.98);
            heat.par_iter_mut()
                .zip(self.prev.par_iter())
                .for_each(|(h, &prev)| *h = *h * (1.0 - p) + prev * p);
        } else if self.prev.len() != n {
            self.agc_init = false;
        }
        self.prev.clone_from(&heat);

        box_blur(
            &heat,
            w,
            h,
            s.detail_radius,
            &mut self.scratch,
            &mut self.blur,
        );

        let (lo, hi) = if s.agc {
            let (mn, mx) = heat
                .par_iter()
                .fold(|| (f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)))
                .reduce(
                    || (f32::MAX, f32::MIN),
                    |(a, b), (c, d)| (a.min(c), b.max(d)),
                );
            const BINS: usize = 1024;
            let span = (mx - mn).max(1e-6);
            let mut hist = [0u32; BINS];
            for &v in &heat {
                let b = (((v - mn) / span) * (BINS - 1) as f32) as usize;
                hist[b.min(BINS - 1)] += 1;
            }
            let pct = |p: f32| {
                let target = (p / 100.0 * n as f32) as u32;
                let mut acc = 0;
                for (i, &c) in hist.iter().enumerate() {
                    acc += c;
                    if acc >= target {
                        return mn + span * i as f32 / (BINS - 1) as f32;
                    }
                }
                mx
            };
            let (tlo, thi) = (pct(s.agc_low_pct), pct(s.agc_high_pct));
            if !self.agc_init {
                self.agc_lo = tlo;
                self.agc_hi = thi;
                self.agc_init = true;
            } else {
                let k = s.agc_speed.clamp(0.001, 1.0);
                self.agc_lo += (tlo - self.agc_lo) * k;
                self.agc_hi += (thi - self.agc_hi) * k;
            }
            (self.agc_lo, self.agc_hi.max(self.agc_lo + 0.02))
        } else if log_floor > 0.0 {
            (log_floor.ln(), (1.0 + log_floor).ln())
        } else {
            (0.0, 1.0)
        };

        const EQ_BINS: usize = 256;
        let inv_span = 1.0 / (hi - lo);
        if s.equalize > 0.0 {
            let mut hist = [0f32; EQ_BINS];
            for &b in &self.blur {
                let t = ((b - lo) * inv_span).clamp(0.0, 1.0);
                hist[(t * (EQ_BINS - 1) as f32) as usize] += 1.0;
            }
            let plateau = s.plateau.max(0.05) * n as f32 / EQ_BINS as f32;
            let mut cdf = [0f32; EQ_BINS];
            let mut acc = 0.0;
            for (c, &hv) in cdf.iter_mut().zip(hist.iter()) {
                acc += hv.min(plateau);
                *c = acc;
            }
            let total = acc.max(1.0);
            if self.eq_lut.len() != EQ_BINS {
                self.eq_lut = cdf.iter().map(|c| c / total).collect();
            } else {
                let k = s.agc_speed.clamp(0.001, 1.0);
                for (l, c) in self.eq_lut.iter_mut().zip(cdf.iter()) {
                    *l += (c / total - *l) * k;
                }
            }
        }

        if self.fpn_cols.len() != w {
            self.fpn_cols = (0..w)
                .map(|x| hash(x as u32 ^ 0xa511_e9b3) * 2.0 - 1.0)
                .collect();
        }
        let frame_seed = self.frame.wrapping_mul(0x9e37_79b9);
        let (cx, cy) = (w as f32 * 0.5, h as f32 * 0.5);
        let inv_r2 = 1.0 / (cx * cx + cy * cy);
        let fpn_cols = &self.fpn_cols;
        let blur = &self.blur;
        let line_px_deg = 0.6 * s.horizon.vfov_deg / h as f32;
        let line = if hz_on && s.horizon.show_line {
            self.horizon.px.as_slice()
        } else {
            &[]
        };
        let lut = &self.eq_lut;
        let eq_amt = if lut.len() == EQ_BINS {
            s.equalize.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let detail_k = 1.0 + s.detail;

        let rgba: Vec<[u8; 4]> = heat
            .par_iter()
            .enumerate()
            .map(|(i, &v)| {
                let (x, y) = (i % w, i / w);
                let base = blur[i];
                let lin = (base - lo) * inv_span;
                let (mut b, mut slope) = (lin, 1.0);
                if eq_amt > 0.0 {
                    let t = lin.clamp(0.0, 1.0) * (EQ_BINS - 1) as f32;
                    let j = (t as usize).min(EQ_BINS - 2);
                    let f = t - j as f32;
                    let eq = lut[j] + (lut[j + 1] - lut[j]) * f;
                    let lut_slope = ((lut[j + 1] - lut[j]) * (EQ_BINS - 1) as f32).clamp(0.5, 3.0);
                    b = lin + (eq - lin) * eq_amt;
                    slope = 1.0 + (lut_slope - 1.0) * eq_amt;
                }
                let mut v = b + (v - base) * inv_span * detail_k * slope;
                v = (v - 0.5) * s.gain + 0.5 + s.level;
                v = v.clamp(0.0, 1.0).powf(s.gamma);
                if s.polarity == Polarity::BlackHot {
                    v = 1.0 - v;
                }
                v += s.noise * (hash(i as u32 ^ frame_seed) * 2.0 - 1.0);
                v += s.fpn * fpn_cols[x];
                let (dx, dy) = (x as f32 - cx, y as f32 - cy);
                v *= 1.0 - s.vignette * (dx * dx + dy * dy) * inv_r2;
                v *= s.brightness.clamp(0.0, 1.0);
                if !line.is_empty() && line[i][2].abs() < line_px_deg {
                    return [255, 0, 0, 255];
                }
                to_rgba8(palette_rgb(s.palette, v.clamp(0.0, 1.0)))
            })
            .collect();

        FlirImage { w, h, rgba }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba16f_downsample() {
        let (w, h) = (32usize, 2usize);
        let pitch = w * 8 + 16;
        let mut buf = vec![0u8; pitch * h];
        let bands = [0.0005f32, 0.0010, 1.0, 4.0];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let b = half::f16::from_f32(bands[x / 8]).to_le_bytes();
                    let i = y * pitch + x * 8 + c * 2;
                    buf[i..i + 2].copy_from_slice(&b);
                }
            }
        }
        let f = downsample_raw(&buf, w, h, pitch, PixelFormat::Rgba16F, w, Crop::default());
        assert_eq!((f.w, f.h), (w, h));
        let v: Vec<f32> = [0, 8, 16, 24].iter().map(|&x| f.rgb[x][0]).collect();

        assert!(v[0] > 0.0 && v[1] > v[0], "shadow levels collapsed: {v:?}");
        assert!((v[2] - 1.0).abs() < 1e-3);
        assert!(v[3] > 1.5, "HDR clipped: {v:?}");
    }

    #[test]
    fn crop_selects_region() {
        let (w, h) = (40usize, 20usize);
        let mut buf = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 20..w {
                buf[(y * w + x) * 4] = 255;
            }
        }
        let crop = Crop {
            x: 0.5,
            y: 0.0,
            w: 0.5,
            h: 1.0,
        };
        let f = downsample_raw(&buf, w, h, w * 4, PixelFormat::Rgba8, 20, crop);
        assert_eq!((f.w, f.h), (20, 20));
        assert!(f.rgb.iter().all(|c| c[0] == 1.0));
    }

    const INK: [f32; 3] = [70.0 / 255.0, 1.0, 110.0 / 255.0];

    fn frame(rgb: Vec<[f32; 3]>, w: usize) -> SourceFrame {
        SourceFrame {
            w,
            h: rgb.len() / w,
            rgb,
        }
    }

    /// Add `k` of the ink to an encoded pixel, in linear light — what the
    /// collimated shader does to the scene behind it.
    fn plus_ink(c: [f32; 3], k: f32) -> [f32; 3] {
        [0, 1, 2].map(|j| (c[j].powf(GAMMA) + k * INK[j].powf(GAMMA)).powf(1.0 / GAMMA))
    }

    #[test]
    fn the_huds_light_on_a_grey_scene_comes_back_out() {
        let scene = [0.3, 0.3, 0.3];
        let src = frame(vec![scene, plus_ink(scene, 0.2), plus_ink(scene, 0.05)], 3);
        let out = reject_ink(&src, INK, 0.0);
        for (i, px) in out.iter().enumerate() {
            for j in 0..3 {
                assert!((px[j] - scene[j]).abs() < 1e-3, "{i}: {px:?}");
            }
        }
    }

    #[test]
    fn ink_on_black_is_black_and_a_blue_sky_is_untouched() {
        let src = frame(vec![plus_ink([0.0; 3], 0.4), [0.2, 0.3, 0.7]], 2);
        let out = reject_ink(&src, INK, 0.0);
        assert!(out[0].iter().all(|&v| v < 1e-3), "{:?}", out[0]);
        for j in 0..3 {
            assert!((out[1][j] - src.rgb[1][j]).abs() < 1e-4, "{:?}", out[1]);
        }
    }

    #[test]
    fn a_clipped_stroke_is_refilled_from_its_neighbours() {
        let (w, h) = (9, 9);
        let scene = [0.4, 0.4, 0.4];
        let mut rgb = vec![scene; w * h];
        rgb[4 * w + 4] = [0.9, 1.0, 0.95]; // a stroke that clipped green
        let out = reject_ink(&frame(rgb, w), INK, 0.35);
        for j in 0..3 {
            assert!(
                (out[4 * w + 4][j] - 0.4).abs() < 1e-3,
                "{:?}",
                out[4 * w + 4]
            );
        }
    }

    /// What MSFS's HUD actually added, flown 13 Oct 2026, linear: bluer
    /// than the gauge's ink.
    fn flown_ink() -> [f32; 3] {
        unit3([0.087, 1.0, 0.327]).unwrap()
    }

    /// A grey scene crossed by strokes of `dir` light (linear, unit) at
    /// `k`, every fourth column — the shape `estimate_ink` learns from.
    fn striped(dir: [f32; 3], k: f32) -> (SourceFrame, [f32; 3]) {
        let scene = [0.35f32; 3];
        let (w, h) = (32, 16);
        let rgb = (0..w * h)
            .map(|i| {
                if (i % w) % 4 == 1 {
                    [0, 1, 2].map(|j| (scene[j].powf(GAMMA) + k * dir[j]).powf(1.0 / GAMMA))
                } else {
                    scene
                }
            })
            .collect();
        (frame(rgb, w), scene)
    }

    #[test]
    fn the_huds_real_colour_is_measured_from_its_strokes() {
        let (src, _) = striped(flown_ink(), 0.4);
        let m = estimate_ink(&src).expect("strokes enough to measure");
        let angle = dot3(m, flown_ink()).clamp(-1.0, 1.0).acos().to_degrees();
        assert!(angle < 1.0, "measured {m:?}, off by {angle}°");
        // A plain grey frame teaches nothing.
        assert!(estimate_ink(&frame(vec![[0.4; 3]; 64], 8)).is_none());
    }

    #[test]
    fn a_stroke_in_the_measured_colour_leaves_no_ghost() {
        let (src, scene) = striped(flown_ink(), 0.4);
        let want = scene[0];
        let worst = |l: &[f32]| l.iter().map(|v| (v - want).abs()).fold(0.0f32, f32::max);
        let nominal_only = ink_free_luminance(&src, INK, None, 0.0);
        let measured = ink_free_luminance(&src, INK, estimate_ink(&src), 0.0);
        assert!(
            worst(&nominal_only) > 0.02,
            "the premise: the nominal ink alone leaves the stroke ({})",
            worst(&nominal_only)
        );
        assert!(worst(&measured) < 2e-3, "ghost left: {}", worst(&measured));
        // And the nominal ink is still taken out with the measured one in
        // the plane.
        let (nominal_src, _) = striped(unit3(INK.map(lin)).unwrap(), 0.4);
        let l = ink_free_luminance(&nominal_src, INK, estimate_ink(&src), 0.0);
        assert!(worst(&l) < 2e-3, "{}", worst(&l));
    }

    #[test]
    fn garbage_ink_leaves_the_frame_alone() {
        let src = frame(vec![[0.1, 0.5, f32::NAN]], 1);
        assert_eq!(reject_ink(&src, [0.0; 3], 0.35).len(), 1);
        assert_eq!(reject_ink(&src, [1.0; 3], 0.35).len(), 1);
        let out = reject_ink(&src, INK, 0.35);
        assert!(out[0].iter().all(|v| v.is_finite()));
    }

    #[test]
    fn the_warp_resamples_the_quad_it_names() {
        // Right half of the frame red, left black; a warp onto the right
        // half sees only red, and one past the frame's edge sees black.
        let (w, h) = (64usize, 32usize);
        let mut buf = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in w / 2..w {
                buf[(y * w + x) * 4] = 255;
            }
        }
        let right =
            Warp::from_corners([(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)], 1.0).unwrap();
        let f = downsample_warped(
            &buf,
            w,
            h,
            w * 4,
            PixelFormat::Rgba8,
            16,
            Crop::default(),
            &right,
        );
        assert_eq!((f.w, f.h), (16, 16));
        assert!(f.rgb.iter().all(|c| (c[0] - 1.0).abs() < 1e-6), "all red");

        let off =
            Warp::from_corners([(1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0)], 0.5).unwrap();
        let f = downsample_warped(
            &buf,
            w,
            h,
            w * 4,
            PixelFormat::Rgba8,
            16,
            Crop::default(),
            &off,
        );
        assert_eq!((f.w, f.h), (16, 8));
        assert!(f.rgb.iter().all(|c| c[0] == 0.0), "off-screen is black");
    }

    #[test]
    fn the_warp_is_relative_to_the_viewport_crop() {
        // A 4-px title bar of white above a black viewport.
        let (w, h) = (32usize, 36usize);
        let mut buf = vec![0u8; w * h * 4];
        buf[..w * 4 * 4].fill(255);
        let crop = Crop {
            x: 0.0,
            y: 4.0 / 36.0,
            w: 1.0,
            h: 32.0 / 36.0,
        };
        let f = downsample_warped(
            &buf,
            w,
            h,
            w * 4,
            PixelFormat::Rgba8,
            16,
            crop,
            &Warp::IDENTITY,
        );
        assert!(
            f.rgb.iter().all(|c| c[0] == 0.0),
            "the title bar is not the scene"
        );
    }

    #[test]
    fn gray_packet_keeps_palette_intensity() {
        let img = FlirImage {
            w: 2,
            h: 1,
            rgba: vec![[10, 200, 5, 255], [90, 90, 90, 255]],
        };
        assert_eq!(img.packet(Encoding::Gray8).data, vec![200, 90]);
        assert_eq!(img.packet(Encoding::Rgba8).data.len(), 8);
    }
}
