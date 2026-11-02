//! Cloud compositing from the colour layer (moved from flir-live).
//!
//! The sim renders volumetric clouds into the nose colour target but they
//! never write depth, so the depth-only scene develops straight through
//! them. Cloud pixels are flagged as bright (relative to a high luminance
//! percentile) AND desaturated (white — excludes blue sky and coloured
//! terrain), and cold cloud radiance at the cloud altitude's air
//! temperature is blended in, obscuring what is behind — which is how a real
//! LWIR sensor sees thick water cloud too.

use flir_env::Environment;
use flir_env::planck::RadianceLut;
use flir_sensor::cpu::ScalarFrame;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloudParams {
    pub enabled: bool,
    /// Brightness, relative to the 98th luminance percentile, where a pixel
    /// starts to count as cloud.
    pub thresh: f32,
    /// Saturation below which a pixel is "white".
    pub sat: f32,
    pub gain: f32,
    /// Altitude whose air temperature the cloud radiates at, metres.
    pub alt_m: f32,
}

impl Default for CloudParams {
    fn default() -> Self {
        Self {
            enabled: true,
            thresh: 0.6,
            sat: 0.15,
            gain: 1.0,
            alt_m: 3000.0,
        }
    }
}

/// Blend cloud radiance into `scene` from linear `rgb` (same size).
pub fn composite(
    scene: &mut ScalarFrame,
    rgb: &[f32],
    env: &Environment,
    lut: &RadianceLut,
    p: &CloudParams,
) {
    let n = scene.w * scene.h;
    if !p.enabled || rgb.len() < n * 3 || scene.data.len() < n {
        return;
    }
    let mut samp: Vec<f32> = rgb
        .chunks_exact(3)
        .step_by(17)
        .map(|c| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])
        .filter(|v| v.is_finite())
        .collect();
    let refl = percentile(&mut samp, 0.98).max(1e-4);
    let l_cloud = lut.lookup(env.air_temp_at(p.alt_m));
    for i in 0..n {
        let (r, g, b) = (rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
        if !(r.is_finite() && g.is_finite() && b.is_finite()) {
            continue;
        }
        let maxc = r.max(g).max(b);
        let minc = r.min(g).min(b);
        let sat = if maxc > 1e-4 { (maxc - minc) / maxc } else { 0.0 };
        let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let bright = smoothstep(p.thresh, p.thresh + 0.35, lum / refl);
        let white = 1.0 - smoothstep(p.sat, p.sat + 0.2, sat);
        let op = (bright * white * p.gain).clamp(0.0, 1.0);
        if op > 0.0 {
            scene.data[i] = op * l_cloud + (1.0 - op) * scene.data[i];
        }
    }
}

/// Smoothstep (Hermite) between `e0` and `e1`.
pub fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0).abs().max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// `p`-quantile of `v` (sorts in place). `p` in [0,1].
pub fn percentile(v: &mut [f32], p: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((v.len() as f32 - 1.0) * p.clamp(0.0, 1.0)).round() as usize;
    v[idx.min(v.len() - 1)]
}
