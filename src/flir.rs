use eframe::egui::{Color32, ColorImage};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::horizon::{HorizonCache, HorizonSettings};

pub struct SourceFrame {
    pub w: usize,
    pub h: usize,
    pub rgb: Vec<[f32; 3]>,
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
    pub polarity: Polarity,
    pub palette: Palette,
    pub noise: f32,
    pub fpn: f32,
    pub vignette: f32,
    pub smooth_display: bool,
    pub show_source: bool,
    pub horizon: HorizonSettings,
    pub capture_hdr: bool,
}

impl Default for FlirSettings {
    fn default() -> Self {
        Self {
            sensor_width: 640,
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
            polarity: Polarity::WhiteHot,
            palette: Palette::Green,
            noise: 0.03,
            fpn: 0.012,
            vignette: 0.3,
            smooth_display: true,
            show_source: false,
            horizon: HorizonSettings::default(),
            capture_hdr: true,
        }
    }
}

#[cfg_attr(not(windows), allow(dead_code))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PixelFormat {
    Rgba8,
    Bgra8,
    Rgba16F,
}

#[cfg_attr(not(windows), allow(dead_code))]
pub fn downsample_raw(
    src: &[u8],
    w: usize,
    h: usize,
    pitch: usize,
    fmt: PixelFormat,
    out_w: usize,
) -> SourceFrame {
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
                    let i = y * pitch + x * 4;
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
                    let i = y * pitch + x * 8;
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

    pub fn process(&mut self, src: &SourceFrame, s: &FlirSettings) -> ColorImage {
        let (w, h) = (src.w, src.h);
        let n = w * h;
        self.frame = self.frame.wrapping_add(1);

        if s.show_source {
            let px = src
                .rgb
                .iter()
                .map(|c| {
                    Color32::from_rgb(
                        (c[0] * 255.0) as u8,
                        (c[1] * 255.0) as u8,
                        (c[2] * 255.0) as u8,
                    )
                })
                .collect();
            return ColorImage::new([w, h], px);
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
        let mut heat: Vec<f32> = src
            .rgb
            .par_iter()
            .enumerate()
            .map(|(i, &[r, g, b])| {
                let hue_gate = ((g - r) / b.max(1e-3) * 4.0 + 0.5).clamp(0.0, 1.0);
                let blueness = (b - r.max(g)).max(0.0) * hue_gate;
                let mut h = wr * r + wg * g + wb * b - sky * blueness;
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

        let pixels: Vec<Color32> = heat
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
                if !line.is_empty() && line[i][2].abs() < line_px_deg {
                    return Color32::RED;
                }
                let c = palette_rgb(s.palette, v.clamp(0.0, 1.0));
                Color32::from_rgb(
                    (c[0].clamp(0.0, 1.0) * 255.0) as u8,
                    (c[1].clamp(0.0, 1.0) * 255.0) as u8,
                    (c[2].clamp(0.0, 1.0) * 255.0) as u8,
                )
            })
            .collect();

        ColorImage::new([w, h], pixels)
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
        let f = downsample_raw(&buf, w, h, pitch, PixelFormat::Rgba16F, w);
        assert_eq!((f.w, f.h), (w, h));
        let v: Vec<f32> = [0, 8, 16, 24].iter().map(|&x| f.rgb[x][0]).collect();

        assert!(v[0] > 0.0 && v[1] > v[0], "shadow levels collapsed: {v:?}");
        assert!((v[2] - 1.0).abs() < 1e-3);
        assert!(v[3] > 1.5, "HDR clipped: {v:?}");
    }
}
