//! CPU reference implementation of the sensor pipeline. Single-threaded and
//! allocation-light; correctness and clarity over speed (the real-time path is
//! the HLSL). `flir-live` runs this on the streamed frames.

use flir_types::Polarity;

use crate::{Sensor, palette_rgb};

/// A single-channel scene-radiance frame (row-major, `w*h`). Values are the
/// scene model's normalised in-band radiance, roughly centred on 1.0 for a
/// 15 °C blackbody but unbounded.
pub struct ScalarFrame {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

/// Output RGBA8 image.
pub struct RgbaFrame {
    pub w: usize,
    pub h: usize,
    pub rgba: Vec<u8>,
}

/// Per-sensor persistent state (AGC smoothing, column FPN pattern, frame
/// counter). Kept across frames by the caller.
pub struct SensorState {
    agc_lo: f32,
    agc_hi: f32,
    agc_init: bool,
    frame: u32,
    fpn_cols: Vec<f32>,
    base: Vec<f32>,
    tmp: Vec<f32>,
}

impl Default for SensorState {
    fn default() -> Self {
        SensorState { agc_lo: 0.0, agc_hi: 1.0, agc_init: false, frame: 0, fpn_cols: Vec::new(), base: Vec::new(), tmp: Vec::new() }
    }
}

impl SensorState {
    pub fn reset(&mut self) {
        *self = SensorState::default();
    }

    /// Run the full pipeline. `sensor` holds the tunables.
    #[allow(clippy::needless_range_loop)]
    pub fn process(&mut self, src: &ScalarFrame, sensor: &Sensor) -> RgbaFrame {
        let p = &sensor.params;
        let (w, h) = (src.w, src.h);
        let n = w * h;
        self.frame = self.frame.wrapping_add(1);

        // 1. AGC: percentile limits (smoothed) blended with plateau histogram EQ.
        let (lo, hi) = if p.agc != 0 {
            let (plo, phi) = percentile_limits(&src.data, p.agc_clip);
            if !self.agc_init {
                self.agc_lo = plo;
                self.agc_hi = phi;
                self.agc_init = true;
            } else {
                let sp = p.agc_speed.clamp(0.02, 1.0); // temporal smoothing speed
                self.agc_lo += (plo - self.agc_lo) * sp;
                self.agc_hi += (phi - self.agc_hi) * sp;
            }
            (self.agc_lo, self.agc_hi)
        } else {
            // Manual: centre on `level`, width from `gain`.
            let c = p.level;
            let half = (1.0 / p.gain.max(1e-3)).max(1e-3);
            (c - half, c + half)
        };
        let span = (hi - lo).max(1e-4);

        // Linear-stretch each pixel; optionally blend plateau-equalized value.
        let mut lin = vec![0.0f32; n];
        for (o, &v) in lin.iter_mut().zip(&src.data) {
            *o = ((v - lo) / span).clamp(0.0, 1.0);
        }
        if p.agc != 0 && p.agc_plateau > 0.0 {
            let eq = plateau_equalize(&lin, 0.02);
            for (o, &e) in lin.iter_mut().zip(&eq) {
                *o += (e - *o) * p.agc_plateau;
            }
        }

        // 2. Base/detail split (box blur), add scaled detail back.
        if p.detail != 0 {
            box_blur(&lin, w, h, 6, &mut self.tmp, &mut self.base);
            for i in 0..n {
                let detail = lin[i] - self.base[i];
                lin[i] = (self.base[i] + detail * (1.0 + p.detail_gain)).clamp(0.0, 1.0);
            }
        }

        // 3. Gamma. (Gain/level already folded into AGC span above.)
        if (p.gamma - 1.0).abs() > 1e-3 {
            let g = p.gamma.max(1e-3);
            for v in &mut lin {
                *v = v.powf(g);
            }
        }

        // 3b. Global output brightness (scales bright midtones too, so it dims an
        // overexposed scene where gamma alone cannot).
        if (p.brightness - 1.0).abs() > 1e-3 {
            let b = p.brightness.max(0.0);
            for v in &mut lin {
                *v = (*v * b).clamp(0.0, 1.0);
            }
        }

        // 4. Noise + FPN + halo + vignette.
        if p.column_fpn != 0 {
            if self.fpn_cols.len() != w {
                self.fpn_cols = (0..w).map(|x| (hash(x as u32 ^ 0x9e37) - 0.5) * 0.02).collect();
            }
            for y in 0..h {
                for x in 0..w {
                    lin[y * w + x] = (lin[y * w + x] + self.fpn_cols[x]).clamp(0.0, 1.0);
                }
            }
        }
        if p.noise != 0 {
            let sigma = p.noise_sigma;
            for (i, v) in lin.iter_mut().enumerate() {
                let g = hash(i as u32 ^ self.frame.wrapping_mul(2654435761)) - 0.5;
                *v = (*v + g * 2.0 * sigma).clamp(0.0, 1.0);
            }
        }
        // Vignette (always on, subtle).
        let (cx, cy) = (w as f32 * 0.5, h as f32 * 0.5);
        let maxr2 = cx * cx + cy * cy;
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let r2 = (dx * dx + dy * dy) / maxr2;
                lin[y * w + x] *= 1.0 - 0.18 * r2;
            }
        }

        // 5. Polarity + palette.
        let invert = sensor.polarity() == Polarity::BlackHot;
        let pal = sensor.palette();
        let mut rgba = vec![0u8; n * 4];
        for i in 0..n {
            let mut v = lin[i].clamp(0.0, 1.0);
            if invert {
                v = 1.0 - v;
            }
            let c = palette_rgb(pal, v);
            rgba[i * 4] = to_u8(c[0]);
            rgba[i * 4 + 1] = to_u8(c[1]);
            rgba[i * 4 + 2] = to_u8(c[2]);
            rgba[i * 4 + 3] = 255;
        }
        RgbaFrame { w, h, rgba }
    }
}

fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Low/high limits at `clip` and `1-clip` percentiles, via a 1024-bin histogram.
fn percentile_limits(data: &[f32], clip: f32) -> (f32, f32) {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in data {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if !lo.is_finite() || hi <= lo {
        return (0.0, 1.0);
    }
    const BINS: usize = 1024;
    let mut hist = [0u32; BINS];
    let span = hi - lo;
    for &v in data {
        if v.is_finite() {
            let b = (((v - lo) / span) * (BINS - 1) as f32) as usize;
            hist[b.min(BINS - 1)] += 1;
        }
    }
    let total: u32 = hist.iter().sum();
    let clip = clip.clamp(0.0, 0.49);
    let want_lo = (total as f32 * clip) as u32;
    let want_hi = (total as f32 * (1.0 - clip)) as u32;
    let (mut acc, mut blo, mut bhi) = (0u32, 0usize, BINS - 1);
    for (i, &c) in hist.iter().enumerate() {
        if acc <= want_lo {
            blo = i;
        }
        if acc <= want_hi {
            bhi = i;
        }
        acc += c;
    }
    let to_v = |b: usize| lo + span * b as f32 / (BINS - 1) as f32;
    (to_v(blo), to_v(bhi.max(blo + 1)))
}

/// Plateau histogram equalization: a CDF with each bin's contribution clipped
/// to a plateau so large uniform regions do not dominate the mapping.
fn plateau_equalize(norm: &[f32], plateau: f32) -> Vec<f32> {
    const BINS: usize = 256;
    let mut hist = [0u32; BINS];
    for &v in norm {
        let b = (v.clamp(0.0, 1.0) * (BINS - 1) as f32) as usize;
        hist[b] += 1;
    }
    let cap = (norm.len() as f32 * plateau).max(1.0) as u32;
    let mut cdf = [0.0f32; BINS];
    let mut acc = 0.0f32;
    let mut total = 0.0f32;
    for &c in &hist {
        total += c.min(cap) as f32;
    }
    let total = total.max(1.0);
    for i in 0..BINS {
        acc += hist[i].min(cap) as f32;
        cdf[i] = acc / total;
    }
    // Linearly interpolate the CDF between adjacent bins. A nearest-bin lookup
    // quantizes the output to BINS discrete levels, which posterizes smooth
    // gradients into visible bands (worst at high plateau blend); interpolation
    // keeps the mapping continuous.
    norm.iter()
        .map(|&v| {
            let f = v.clamp(0.0, 1.0) * (BINS - 1) as f32;
            let b = f.floor() as usize;
            let frac = f - b as f32;
            let c0 = cdf[b];
            let c1 = cdf[(b + 1).min(BINS - 1)];
            c0 + (c1 - c0) * frac
        })
        .collect()
}

fn box_blur(src: &[f32], w: usize, h: usize, r: usize, tmp: &mut Vec<f32>, out: &mut Vec<f32>) {
    tmp.resize(w * h, 0.0);
    out.resize(w * h, 0.0);
    let r = r.max(1) as isize;
    let inv = 1.0 / (2 * r + 1) as f32;
    // Horizontal
    for y in 0..h {
        let s = &src[y * w..(y + 1) * w];
        let at = |x: isize| s[x.clamp(0, w as isize - 1) as usize];
        let mut acc: f32 = (-r..=r).map(at).sum();
        for x in 0..w as isize {
            tmp[y * w + x as usize] = acc * inv;
            acc += at(x + r + 1) - at(x - r);
        }
    }
    // Vertical
    for x in 0..w {
        let at = |y: isize| tmp[y.clamp(0, h as isize - 1) as usize * w + x];
        let mut acc: f32 = (-r..=r).map(at).sum();
        for y in 0..h as isize {
            out[y as usize * w + x] = acc * inv;
            acc += at(y + r + 1) - at(y - r);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(w: usize, h: usize) -> ScalarFrame {
        let data = (0..w * h).map(|i| (i % w) as f32 / w as f32).collect();
        ScalarFrame { w, h, data }
    }

    #[test]
    fn agc_stretches_to_full_range() {
        // A low-contrast input (0.4..0.6) should fill the output range.
        let w = 64;
        let h = 8;
        let data: Vec<f32> = (0..w * h).map(|i| 0.4 + 0.2 * (i % w) as f32 / w as f32).collect();
        let src = ScalarFrame { w, h, data };
        let mut st = SensorState::default();
        let mut params = flir_types::SensorParams::default();
        params.detail = 0;
        params.noise = 0;
        params.column_fpn = 0;
        let out = st.process(&src, &Sensor::new(params));
        let maxg = out.rgba.chunks(4).map(|p| p[1]).max().unwrap();
        let ming = out.rgba.chunks(4).map(|p| p[1]).min().unwrap();
        assert!(maxg > 200, "max green {maxg}");
        assert!(ming < 80, "min green {ming}");
    }

    #[test]
    fn black_hot_inverts() {
        let src = ramp(64, 4);
        let mut params = flir_types::SensorParams::default();
        params.agc = 0;
        params.gain = 1.0;
        params.level = 0.5;
        params.detail = 0;
        params.noise = 0;
        params.column_fpn = 0;
        params.palette_u8 = flir_types::Palette::Gray as u8;
        let mut a = SensorState::default();
        let white = a.process(&src, &Sensor::new(params));
        params.polarity_u8 = Polarity::BlackHot as u8;
        let mut b = SensorState::default();
        let black = b.process(&src, &Sensor::new(params));
        // Right edge is hottest: white-hot bright, black-hot dark.
        let idx = (63 * 4) as usize;
        assert!(white.rgba[idx] > black.rgba[idx]);
    }
}
