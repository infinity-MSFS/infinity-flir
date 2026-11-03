//! Band-integrated Planck radiance and its inverse, precomputed as a lookup.
//!
//! The scene model needs `B(T)` — the blackbody radiance integrated over the
//! sensor band — many times per pixel. We tabulate it over a plausible scene
//! temperature range and interpolate. The absolute radiometric scale is
//! arbitrary (the sensor model's AGC normalises it); what matters is the
//! *shape*: how steeply radiance rises with temperature in each band, which
//! sets scene contrast.

use flir_types::Band;

/// Planck's law spectral radiance (W·m⁻²·sr⁻¹·µm⁻¹) at wavelength `lambda_um`
/// and temperature `t_k` (kelvin).
pub fn spectral_radiance(lambda_um: f64, t_k: f64) -> f64 {
    // c1 = 2hc², c2 = hc/k, in units matched to µm and W/(m²·sr·µm).
    const C1: f64 = 1.191_042e8; // W·µm⁴/(m²·sr·µm)  → gives W/(m²·sr·µm)
    const C2: f64 = 1.438_777e4; // µm·K
    let l5 = lambda_um.powi(5);
    let x = C2 / (lambda_um * t_k);
    C1 / (l5 * (x.exp() - 1.0))
}

/// Integrate spectral radiance over a band by Simpson's rule.
pub fn band_radiance(lo_um: f64, hi_um: f64, t_k: f64, steps: usize) -> f64 {
    let steps = steps.max(2) & !1; // even
    let h = (hi_um - lo_um) / steps as f64;
    let mut sum = spectral_radiance(lo_um, t_k) + spectral_radiance(hi_um, t_k);
    for i in 1..steps {
        let l = lo_um + i as f64 * h;
        sum += spectral_radiance(l, t_k) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    sum * h / 3.0
}

/// Band limits in micrometres.
pub fn band_limits(band: Band) -> (f64, f64) {
    match band {
        Band::Lwir => (8.0, 14.0),
        Band::Mwir => (3.0, 5.0),
    }
}

/// Lowest and highest tabulated temperature (°C). Covers cold high-altitude
/// cloud tops through hot tarmac / exhaust-warmed surfaces.
pub const T_MIN_C: f32 = -80.0;
pub const T_MAX_C: f32 = 150.0;
const LUT_N: usize = 256;

/// A band-radiance lookup table, normalised so `lookup(15 °C) == 1.0` to keep
/// downstream values in a convenient range.
pub struct RadianceLut {
    band: Band,
    values: [f32; LUT_N],
    norm: f32,
}

impl RadianceLut {
    pub fn build(band: Band) -> RadianceLut {
        let (lo, hi) = band_limits(band);
        let mut values = [0.0f32; LUT_N];
        for (i, v) in values.iter_mut().enumerate() {
            let t_c = T_MIN_C + (T_MAX_C - T_MIN_C) * i as f32 / (LUT_N - 1) as f32;
            let t_k = t_c as f64 + 273.15;
            *v = band_radiance(lo, hi, t_k, 24) as f32;
        }
        let norm = {
            let r15 = band_radiance(lo, hi, 288.15, 24) as f32;
            if r15 > 0.0 { 1.0 / r15 } else { 1.0 }
        };
        RadianceLut { band, values, norm }
    }

    pub fn band(&self) -> Band {
        self.band
    }

    /// Normalised band radiance at temperature `t_c` (°C), linear-interpolated.
    pub fn lookup(&self, t_c: f32) -> f32 {
        let f = (t_c - T_MIN_C) / (T_MAX_C - T_MIN_C) * (LUT_N - 1) as f32;
        let f = f.clamp(0.0, (LUT_N - 1) as f32);
        let i = f.floor() as usize;
        let frac = f - i as f32;
        let a = self.values[i];
        let b = self.values[(i + 1).min(LUT_N - 1)];
        (a + (b - a) * frac) * self.norm
    }

    /// The raw table (normalised) for uploading to a GPU 1-D texture.
    pub fn normalised_table(&self) -> [f32; LUT_N] {
        let mut out = self.values;
        for v in &mut out {
            *v *= self.norm;
        }
        out
    }

    pub const LEN: usize = LUT_N;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radiance_rises_with_temperature() {
        let lut = RadianceLut::build(Band::Lwir);
        assert!(lut.lookup(40.0) > lut.lookup(0.0));
        assert!(lut.lookup(0.0) > lut.lookup(-40.0));
    }

    #[test]
    fn normalised_at_15c() {
        let lut = RadianceLut::build(Band::Lwir);
        assert!(
            (lut.lookup(15.0) - 1.0).abs() < 0.02,
            "{}",
            lut.lookup(15.0)
        );
    }

    #[test]
    fn mwir_steeper_than_lwir() {
        // Wien: shorter wavelengths have steeper radiance-vs-T near ambient,
        // so MWIR contrast per degree exceeds LWIR.
        let lwir = RadianceLut::build(Band::Lwir);
        let mwir = RadianceLut::build(Band::Mwir);
        let lwir_ratio = lwir.lookup(40.0) / lwir.lookup(10.0);
        let mwir_ratio = mwir.lookup(40.0) / mwir.lookup(10.0);
        assert!(
            mwir_ratio > lwir_ratio,
            "mwir {mwir_ratio} lwir {lwir_ratio}"
        );
    }
}
