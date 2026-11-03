//! Solar position (NOAA Solar Calculator algorithm).
//!
//! Used when the engine does not expose a sun direction. Inputs are UTC and
//! geodetic position; outputs are elevation and azimuth, and a world-space
//! direction *to* the sun in the sim's local ENU frame (x east, y up, z north,
//! matching how the scene model consumes `sun_dir`).
//!
//! Accuracy is ~0.01° over 1950–2050, far finer than the thermal model needs.
//! Reference: NOAA GML Solar Calculator spreadsheet equations.

use core::f64::consts::PI;

const DEG: f64 = PI / 180.0;

#[derive(Clone, Copy, Debug)]
pub struct SolarPosition {
    /// Degrees above the horizon (negative at night). Not refraction-corrected
    /// beyond NOAA's standard term.
    pub elevation_deg: f64,
    /// Degrees clockwise from true north.
    pub azimuth_deg: f64,
}

impl SolarPosition {
    /// Unit vector pointing *toward* the sun in local ENU (east, up, north).
    pub fn direction_enu(&self) -> [f32; 3] {
        let el = self.elevation_deg * DEG;
        let az = self.azimuth_deg * DEG;
        let horiz = libm_cos(el);
        let east = horiz * libm_sin(az);
        let north = horiz * libm_cos(az);
        let up = libm_sin(el);
        [east as f32, up as f32, north as f32]
    }
}

/// Julian day from a Unix UTC timestamp (seconds).
fn julian_day(unix_s: f64) -> f64 {
    // Unix epoch = JD 2440587.5.
    unix_s / 86400.0 + 2440587.5
}

/// Compute the solar position for a UTC time and position.
pub fn solar_position(unix_s: f64, lat_deg: f64, lon_deg: f64) -> SolarPosition {
    let jd = julian_day(unix_s);
    // Julian century since J2000.0.
    let t = (jd - 2451545.0) / 36525.0;

    let geom_mean_long = norm360(280.46646 + t * (36000.76983 + t * 0.0003032));
    let geom_mean_anom = 357.52911 + t * (35999.05029 - 0.0001537 * t);
    let eccent = 0.016708634 - t * (0.000042037 + 0.0000001267 * t);

    let m = geom_mean_anom * DEG;
    let sun_eq_ctr = libm_sin(m) * (1.914602 - t * (0.004817 + 0.000014 * t))
        + libm_sin(2.0 * m) * (0.019993 - 0.000101 * t)
        + libm_sin(3.0 * m) * 0.000289;
    let true_long = geom_mean_long + sun_eq_ctr;
    let app_long = true_long - 0.00569 - 0.00478 * libm_sin((125.04 - 1934.136 * t) * DEG);

    let mean_obliq =
        23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0;
    let obliq_corr = mean_obliq + 0.00256 * libm_cos((125.04 - 1934.136 * t) * DEG);

    let decl = libm_asin(libm_sin(obliq_corr * DEG) * libm_sin(app_long * DEG)) / DEG;

    // Equation of time (minutes).
    let y = libm_tan(obliq_corr / 2.0 * DEG);
    let y = y * y;
    let l0 = geom_mean_long * DEG;
    let eot = 4.0 / DEG
        * (y * libm_sin(2.0 * l0) - 2.0 * eccent * libm_sin(m)
            + 4.0 * eccent * y * libm_sin(m) * libm_cos(2.0 * l0)
            - 0.5 * y * y * libm_sin(4.0 * l0)
            - 1.25 * eccent * eccent * libm_sin(2.0 * m));

    // True solar time (minutes) from the UTC fraction of the day.
    let day_frac = (unix_s / 86400.0).rem_euclid(1.0);
    let minutes_utc = day_frac * 1440.0;
    let true_solar_time = (minutes_utc + eot + 4.0 * lon_deg).rem_euclid(1440.0);

    let hour_angle = if true_solar_time / 4.0 < 0.0 {
        true_solar_time / 4.0 + 180.0
    } else {
        true_solar_time / 4.0 - 180.0
    };

    let lat = lat_deg * DEG;
    let decl_r = decl * DEG;
    let ha = hour_angle * DEG;
    let cos_zenith =
        libm_sin(lat) * libm_sin(decl_r) + libm_cos(lat) * libm_cos(decl_r) * libm_cos(ha);
    let cos_zenith = cos_zenith.clamp(-1.0, 1.0);
    let zenith = libm_acos(cos_zenith);
    let elevation = 90.0 - zenith / DEG;

    // Azimuth (clockwise from north).
    let az_denom = libm_cos(lat) * libm_sin(zenith);
    let azimuth = if az_denom.abs() > 1e-9 {
        let mut a = libm_acos(((libm_sin(lat) * cos_zenith) - libm_sin(decl_r)) / az_denom) / DEG;
        if hour_angle > 0.0 {
            a = norm360(a + 180.0);
        } else {
            a = norm360(540.0 - a);
        }
        a
    } else if lat_deg > 0.0 {
        180.0
    } else {
        0.0
    };

    // Atmospheric-refraction correction to elevation (NOAA approximation).
    let refr = refraction_deg(elevation);
    SolarPosition {
        elevation_deg: elevation + refr,
        azimuth_deg: azimuth,
    }
}

fn refraction_deg(elev: f64) -> f64 {
    if elev > 85.0 {
        0.0
    } else {
        let te = libm_tan(elev * DEG);
        let r = if elev > 5.0 {
            58.1 / te - 0.07 / (te * te * te) + 0.000086 / (te.powi(5))
        } else if elev > -0.575 {
            1735.0 + elev * (-518.2 + elev * (103.4 + elev * (-12.79 + elev * 0.711)))
        } else {
            -20.772 / te
        };
        r / 3600.0
    }
}

fn norm360(x: f64) -> f64 {
    x.rem_euclid(360.0)
}

// --- Minimal libm-style shims (f64) so this stays std-only without a math dep.
// std's f64 methods are available; these just name them for readability.
#[inline]
fn libm_sin(x: f64) -> f64 {
    x.sin()
}
#[inline]
fn libm_cos(x: f64) -> f64 {
    x.cos()
}
#[inline]
fn libm_tan(x: f64) -> f64 {
    x.tan()
}
#[inline]
fn libm_asin(x: f64) -> f64 {
    x.clamp(-1.0, 1.0).asin()
}
#[inline]
fn libm_acos(x: f64) -> f64 {
    x.clamp(-1.0, 1.0).acos()
}

#[cfg(test)]
mod tests {
    use super::*;

    // 2024-06-21 12:00 UTC at (51.48N, 0W) Greenwich: sun high in the south.
    #[test]
    fn summer_noon_greenwich() {
        // 2024-06-21T12:00:00Z
        let unix = 1_718_971_200.0;
        let p = solar_position(unix, 51.4769, 0.0);
        assert!(
            p.elevation_deg > 55.0 && p.elevation_deg < 63.0,
            "elev {}",
            p.elevation_deg
        );
        assert!((p.azimuth_deg - 180.0).abs() < 25.0, "az {}", p.azimuth_deg);
        let d = p.direction_enu();
        assert!(d[1] > 0.0, "sun should be above horizon");
    }

    #[test]
    fn midnight_is_below_horizon() {
        // 2024-12-21T00:00:00Z at Greenwich.
        let unix = 1_734_739_200.0;
        let p = solar_position(unix, 51.4769, 0.0);
        assert!(p.elevation_deg < 0.0, "elev {}", p.elevation_deg);
    }
}
