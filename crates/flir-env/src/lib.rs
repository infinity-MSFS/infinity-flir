//! `flir-env` — the [`Environment`] type and the derived atmospheric and solar
//! values the thermal scene model consumes.
//!
//! **Pure computation, no SimConnect.** The host (Infinity Manager) reads the
//! SimVars and pushes raw values in via [`flir_types::EnvironmentWire`]; this
//! crate turns them into the quantities the scene model needs: air temperature
//! at any altitude, humidity (measured or estimated), atmospheric extinction,
//! and a sun direction (engine-provided when available, else NOAA).
//!
//! The exact SimVar name for each field is documented in
//! `docs/environment-simvars.md` so the Manager knows what to read.

pub mod atmos;
pub mod planck;
pub mod solar;

use flir_types::{Band, EnvironmentWire};

/// Precipitation state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precip {
    None,
    Rain,
    Snow,
}

impl Precip {
    fn from_u8(v: u8) -> Precip {
        match v {
            1 => Precip::Rain,
            2 => Precip::Snow,
            _ => Precip::None,
        }
    }
}

/// Fully-resolved environment: raw host values plus everything derived once per
/// update so the per-pixel shader path stays cheap.
#[derive(Clone, Copy, Debug)]
pub struct Environment {
    pub ambient_temp_c: f32,
    pub pressure_hpa: f32,
    /// Relative humidity %, always populated (measured, from dew point, or
    /// estimated from visibility — see `humidity_estimated`).
    pub humidity_pct: f32,
    pub humidity_estimated: bool,
    pub visibility_m: f32,
    pub precip: Precip,
    pub precip_rate: f32,
    pub in_cloud: bool,
    pub wind_dir_deg: f32,
    pub wind_speed_ms: f32,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub alt_m: f32,
    pub ground_alt_m: f32,
    pub utc_unix_s: f64,

    // Derived:
    /// Unit vector toward the sun, local ENU (east, up, north).
    pub sun_dir_enu: [f32; 3],
    pub sun_elevation_deg: f32,
    pub sun_from_engine: bool,
    /// Precipitable-water scalar (0..1) for path radiance.
    pub water_loading: f32,
}

impl Environment {
    /// Build from the wire form the host pushes. Fills every derived value.
    pub fn from_wire(w: &EnvironmentWire) -> Environment {
        let precip = Precip::from_u8(w.precip_kind);
        let in_cloud = w.in_cloud != 0;

        // Humidity: measured, else from dew point, else estimated.
        let (humidity_pct, humidity_estimated) = if w.humidity_pct.is_finite() {
            (w.humidity_pct.clamp(0.0, 100.0), false)
        } else if w.dewpoint_c.is_finite() && w.ambient_temp_c.is_finite() {
            (
                atmos::humidity_from_dewpoint(w.dewpoint_c, w.ambient_temp_c),
                false,
            )
        } else {
            (
                atmos::humidity_from_visibility(w.visibility_m, w.precip_rate),
                true,
            )
        };

        let water_loading = atmos::water_loading(humidity_pct, w.precip_rate, in_cloud);

        // Sun: engine value if finite, else NOAA.
        let (sun_dir_enu, sun_elevation_deg, sun_from_engine) =
            if w.sun_dir.iter().all(|c| c.is_finite()) && w.sun_dir.iter().any(|&c| c != 0.0) {
                let d = normalise(w.sun_dir);
                (d, d[1].clamp(-1.0, 1.0).asin().to_degrees(), true)
            } else {
                let p = solar::solar_position(w.utc_unix_s, w.lat_deg, w.lon_deg);
                (p.direction_enu(), p.elevation_deg as f32, false)
            };

        Environment {
            ambient_temp_c: w.ambient_temp_c,
            pressure_hpa: w.pressure_hpa,
            humidity_pct,
            humidity_estimated,
            visibility_m: w.visibility_m,
            precip,
            precip_rate: w.precip_rate,
            in_cloud,
            wind_dir_deg: w.wind_dir_deg,
            wind_speed_ms: w.wind_speed_ms,
            lat_deg: w.lat_deg,
            lon_deg: w.lon_deg,
            alt_m: w.alt_m,
            ground_alt_m: w.ground_alt_m,
            utc_unix_s: w.utc_unix_s,
            sun_dir_enu,
            sun_elevation_deg,
            sun_from_engine,
            water_loading,
        }
    }

    /// Air temperature (°C) at a pixel's altitude via the lapse rate.
    pub fn air_temp_at(&self, pixel_alt_m: f32) -> f32 {
        atmos::air_temp_c(self.ambient_temp_c, self.alt_m, pixel_alt_m)
    }

    /// Atmospheric extinction coefficient (per metre) for a band.
    pub fn extinction_per_m(&self, band: Band) -> f32 {
        atmos::extinction_per_m(band, self.water_loading, self.visibility_m)
    }

    /// Insolation warmth factor in [−0.2, 1].
    pub fn diurnal_warmth(&self) -> f32 {
        atmos::diurnal_warmth(self.sun_elevation_deg)
    }

    /// Whether it is effectively daytime (sun above the horizon).
    pub fn is_day(&self) -> bool {
        self.sun_elevation_deg > 0.0
    }
}

fn normalise(v: [f32; 3]) -> [f32; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n > 1e-6 {
        [v[0] / n, v[1] / n, v[2] / n]
    } else {
        [0.0, 1.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wire() -> EnvironmentWire {
        EnvironmentWire {
            ambient_temp_c: 15.0,
            pressure_hpa: 1013.0,
            humidity_pct: f32::NAN,
            dewpoint_c: f32::NAN,
            visibility_m: 10_000.0,
            precip_kind: 0,
            in_cloud: 0,
            _pad: [0; 2],
            precip_rate: 0.0,
            wind_dir_deg: 0.0,
            wind_speed_ms: 0.0,
            lat_deg: 51.4769,
            lon_deg: 0.0,
            alt_m: 1000.0,
            ground_alt_m: 0.0,
            utc_unix_s: 1_718_971_200.0, // 2024-06-21T12:00Z
            sun_dir: [f32::NAN; 3],
        }
    }

    #[test]
    fn humidity_estimated_when_absent() {
        let e = Environment::from_wire(&wire());
        assert!(e.humidity_estimated);
        assert!(e.humidity_pct > 0.0);
    }

    #[test]
    fn sun_from_noaa_when_absent() {
        let e = Environment::from_wire(&wire());
        assert!(!e.sun_from_engine);
        assert!(e.is_day());
    }

    #[test]
    fn engine_sun_overrides() {
        let mut w = wire();
        w.sun_dir = [0.0, 1.0, 0.0];
        let e = Environment::from_wire(&w);
        assert!(e.sun_from_engine);
        assert!((e.sun_elevation_deg - 90.0).abs() < 0.1);
    }
}
