//! Atmospheric derived values: lapse rate, humidity estimation, and the
//! band-dependent atmospheric extinction coefficient used by the scene model's
//! path-radiance term `τ = exp(−k·range)`.
//!
//! These are engineering approximations tuned for a believable LWIR/MWIR look,
//! not a radiative-transfer solver. Every constant is documented so it can be
//! re-tuned in `flir-live`.

use flir_types::Band;

/// ISA tropospheric lapse rate: temperature falls 6.5 °C per km of altitude.
pub const LAPSE_RATE_C_PER_M: f32 = 0.0065;

/// Air temperature at a given altitude, from the ambient temperature measured
/// at the aircraft. `pixel_alt_m` and `aircraft_alt_m` are geometric altitudes.
pub fn air_temp_c(ambient_c: f32, aircraft_alt_m: f32, pixel_alt_m: f32) -> f32 {
    ambient_c + (aircraft_alt_m - pixel_alt_m) * LAPSE_RATE_C_PER_M
}

/// Saturation vapour pressure (hPa) over water at temperature `t_c`
/// (Magnus-Tetens).
pub fn sat_vapour_pressure_hpa(t_c: f32) -> f32 {
    6.1094 * (17.625 * t_c / (t_c + 243.04)).exp()
}

/// Relative humidity (%) from a dew point, both °C.
pub fn humidity_from_dewpoint(dewpoint_c: f32, temp_c: f32) -> f32 {
    (100.0 * sat_vapour_pressure_hpa(dewpoint_c) / sat_vapour_pressure_hpa(temp_c)).clamp(0.0, 100.0)
}

/// Estimate relative humidity (%) when neither humidity nor dew point is
/// available, from visibility and precipitation. Documented fallback: low
/// visibility and active precipitation imply near-saturation; clear long-range
/// visibility implies a drier column. This is a heuristic, surfaced as such.
pub fn humidity_from_visibility(visibility_m: f32, precip_rate: f32) -> f32 {
    let vis = visibility_m.max(1.0);
    // 50 km clear → ~35%; 1 km haze → ~95%; log-shaped between.
    let base = 95.0 - 13.0 * (vis / 1000.0).log10().clamp(0.0, 4.6);
    let wet = (precip_rate * 8.0).clamp(0.0, 25.0);
    (base + wet).clamp(10.0, 100.0)
}

/// Precipitable-water-like scalar (0..1) driving IR extinction: more water
/// vapour + liquid → more absorption/emission along the path.
pub fn water_loading(humidity_pct: f32, precip_rate: f32, in_cloud: bool) -> f32 {
    let h = (humidity_pct / 100.0).clamp(0.0, 1.0);
    let p = (precip_rate / 10.0).clamp(0.0, 1.0);
    let c = if in_cloud { 1.0 } else { 0.0 };
    (0.55 * h + 0.35 * p + 0.5 * c).min(1.0)
}

/// Atmospheric extinction coefficient `k` (per metre) for the band. Larger in
/// MWIR-humid and LWIR-fog conditions. Combined with slant range as
/// `τ = exp(−k·range)`.
pub fn extinction_per_m(band: Band, water: f32, visibility_m: f32) -> f32 {
    // Baseline clear-air LWIR extinction ~ 1/20km; MWIR a touch less in the
    // 3–5 µm window but hit harder by water vapour bands.
    let (base, water_gain) = match band {
        Band::Lwir => (1.0 / 20_000.0, 1.0 / 6_000.0),
        Band::Mwir => (1.0 / 25_000.0, 1.0 / 4_000.0),
    };
    // Visibility (a visible-band quantity) still tracks aerosol/fog that scatter
    // IR; fold it in weakly.
    let vis_term = (1.0 / visibility_m.max(200.0)) * 0.15;
    base + water * water_gain + vis_term
}

/// Diurnal phase in [0,1): 0 at solar midnight, 0.5 at solar noon. Derived from
/// the sun elevation's recent trajectory is ideal, but a time-of-day proxy is
/// enough for the surface-temperature model; here we use the sun elevation
/// directly (callers pass it) mapped through a smooth curve.
pub fn diurnal_warmth(sun_elevation_deg: f32) -> f32 {
    // Surfaces lag the sun: peak warmth mid-afternoon, min before dawn. We model
    // the *insolation* with elevation and let the scene model apply thermal
    // lag per material. Here just return clamped sin(elev).
    (sun_elevation_deg.to_radians().sin()).clamp(-0.2, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lapse_cools_with_altitude() {
        // Pixel 1000 m below the aircraft is warmer.
        let t = air_temp_c(15.0, 2000.0, 1000.0);
        assert!((t - 21.5).abs() < 0.01, "{t}");
    }

    #[test]
    fn humidity_monotonic_in_visibility() {
        let clear = humidity_from_visibility(50_000.0, 0.0);
        let haze = humidity_from_visibility(1_000.0, 0.0);
        assert!(haze > clear);
        assert!(clear >= 10.0 && haze <= 100.0);
    }

    #[test]
    fn extinction_increases_with_water() {
        let dry = extinction_per_m(Band::Lwir, 0.0, 20_000.0);
        let wet = extinction_per_m(Band::Lwir, 1.0, 20_000.0);
        assert!(wet > dry);
    }
}
