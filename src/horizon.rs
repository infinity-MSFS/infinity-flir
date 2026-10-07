//! To match the real FLIR image more closely, we will keep track of the horizon to keep the sky darker in the output image

use serde::{Deserialize, Serialize};

const EARTH_RADIUS_M: f32 = 6_371_000.0;
const FT_TO_M: f32 = 0.03048;

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(default)]
pub struct HorizonSettings {
    pub enabled: bool,
    pub pitch_deg: f32,
    pub bank_deg: f32,
    /// vertical FOV of the captured image
    pub vfov_deg: f32,
    pub altitude_ft: f32,
    pub ground_temp: f32,
    pub horizon_temp: f32,
    pub zenith_temp: f32,
    /// elevation over which the sky cools from horizon to zenith temperature
    pub sky_falloff_deg: f32,
    /// range at which ground contrast falls to 1/e
    pub atten_km: f32,
    pub ground_vis: f32,
    pub sky_vis: f32,
    pub show_line: bool,
}

impl Default for HorizonSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            pitch_deg: 0.0,
            bank_deg: 0.0,
            vfov_deg: 50.0,
            altitude_ft: 10_000.0,
            ground_temp: 0.35,
            horizon_temp: 0.22,
            zenith_temp: 0.02,
            sky_falloff_deg: 12.0,
            atten_km: 60.0,
            ground_vis: 1.0,
            sky_vis: 0.4,
            show_line: false,
        }
    }
}

#[derive(Default)]
pub struct HorizonCache {
    key: Option<(HorizonSettings, usize, usize)>,
    pub px: Vec<[f32; 3]>,
}

impl HorizonCache {
    pub fn get(&mut self, s: &HorizonSettings, w: usize, h: usize) -> &[[f32; 3]] {
        let key = (s.clone(), w, h);
        if self.key.as_ref() != Some(&key) {
            self.px = compute(s, w, h);
            self.key = Some(key);
        }
        &self.px
    }
}

fn compute(s: &HorizonSettings, w: usize, h: usize) -> Vec<[f32; 3]> {
    use rayon::prelude::*;

    let (pitch, bank) = (s.pitch_deg.to_radians(), s.bank_deg.to_radians());
    let up = [
        -bank.sin() * pitch.cos(),
        bank.cos() * pitch.cos(),
        pitch.sin(),
    ];
    let f = (h as f32 * 0.5) / (s.vfov_deg.to_radians() * 0.5).tan();
    let (cx, cy) = (w as f32 * 0.5, h as f32 * 0.5);

    let alt = (s.altitude_ft * FT_TO_M).max(3.0);
    let r_obs = EARTH_RADIUS_M + alt;
    let dip = (EARTH_RADIUS_M / r_obs).acos();
    let atten_m = (s.atten_km * 1000.0).max(100.0);
    let falloff = s.sky_falloff_deg.max(0.1);

    (0..w * h)
        .into_par_iter()
        .map(|i| {
            let (x, y) = ((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
            let r = [x - cx, cy - y, f];
            let len = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
            let sin_elev = (r[0] * up[0] + r[1] * up[1] + r[2] * up[2]) / len;
            let elev = sin_elev.clamp(-1.0, 1.0).asin();
            let e = elev + dip;
            let e_deg = e.to_degrees();

            let bg = if e >= 0.0 {
                s.zenith_temp + (s.horizon_temp - s.zenith_temp) * (-e_deg / falloff).exp()
            } else {
                let d = -elev;
                let disc =
                    (EARTH_RADIUS_M * EARTH_RADIUS_M - r_obs * r_obs * d.cos().powi(2)).max(0.0);
                let range = r_obs * d.sin() - disc.sqrt();
                let tau = (-range / atten_m).exp();
                s.ground_temp * tau + s.horizon_temp * (1.0 - tau)
            };
            let sky_w = ((e_deg + 0.3) / 0.6).clamp(0.0, 1.0);
            [bg, sky_w, e_deg]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn horizon_row(s: &HorizonSettings, w: usize, h: usize, x: usize) -> usize {
        let px = compute(s, w, h);
        (0..h)
            .min_by(|&a, &b| px[a * w + x][2].abs().total_cmp(&px[b * w + x][2].abs()))
            .unwrap()
    }

    #[test]
    fn attitude_moves_horizon_the_right_way() {
        let (w, h) = (200, 100);
        let mut s = HorizonSettings {
            vfov_deg: 40.0,
            altitude_ft: 10.0,
            ..Default::default()
        };
        let level = horizon_row(&s, w, h, w / 2);
        assert!((level as i32 - 50).abs() <= 1, "level horizon at {level}");

        s.pitch_deg = -10.0;
        assert!(horizon_row(&s, w, h, w / 2) < level);
        s.pitch_deg = 10.0;
        assert!(horizon_row(&s, w, h, w / 2) > level);

        s.pitch_deg = 0.0;
        s.bank_deg = 20.0;
        assert!(horizon_row(&s, w, h, w - 10) < horizon_row(&s, w, h, 10));

        s.bank_deg = 0.0;
        s.altitude_ft = 37_000.0;
        let px = compute(&s, w, h);
        let centre_e = px[(h / 2) * w + w / 2][2];
        assert!((centre_e - 3.4).abs() < 0.3, "dip {centre_e}");
    }
}
