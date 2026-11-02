//! The headless FLIR develop — `flir-live`'s per-frame path without egui, for
//! a production consumer (the Infinity Manager).
//!
//! ```text
//! viewports frame ──► depth (R32F) ─┐
//!                 └─► colour (R11G11B10 / RGBA16F) ─► clouds ─┐
//! EnvironmentWire ──► Environment ──► depth_to_scene ─────────►┴─► SensorState ─► gray
//!                                                       crop/resample ◄──┘
//! ```
//!
//! * [`Pipeline`] holds everything that must persist across frames (the
//!   sensor's AGC/FPN state, the radiance tables). Develop ONCE per new
//!   stream frame: every call advances the AGC and the noise frame counter.
//! * [`decode`] turns the stream's raw layer bytes into floats.
//! * [`clouds`] is the colour-buffer cloud composite (clouds never write
//!   depth, so the depth-only scene would see straight through them).
//! * [`Gray::crop_square`] cuts the centre square and resamples it — a nose
//!   camera whose VERTICAL fov is a square display's field gives exactly
//!   that field in its centre `h × h` (square pixels), so this is the whole
//!   projection a collimated HUD needs.
//!
//! Pure std: no viewports, no OS calls.

pub mod clouds;
pub mod decode;

use flir_env::Environment;
use flir_env::planck::RadianceLut;
use flir_scene::cpu::depth_to_scene;
use flir_sensor::Sensor;
use flir_sensor::cpu::{RgbaFrame, SensorState};
use flir_types::{Band, EnvironmentWire, Palette, SensorParams};

pub use clouds::CloudParams;

/// The tuned default sensor look (flir-live's preset): clip 0.03, plateau
/// 0.33, gamma 5.5, brightness 0.52, strong detail enhancement, no noise.
pub fn preset_params() -> SensorParams {
    SensorParams {
        agc: 1,
        agc_clip: 0.030,
        agc_plateau: 0.33,
        agc_speed: 1.0,
        gamma: 5.5,
        brightness: 0.52,
        detail: 1,
        detail_gain: 2.75,
        noise: 1,
        noise_sigma: 0.0,
        column_fpn: 1,
        ..SensorParams::default()
    }
}

/// [`preset_params`] for a display that tints the picture itself (a HUD):
/// GRAY palette, so channel 0 is the intensity; AGC smoothed over a few
/// frames so a streamed picture does not pump.
pub fn hud_params() -> SensorParams {
    SensorParams {
        palette_u8: Palette::Gray as u8,
        agc_speed: 0.35,
        ..preset_params()
    }
}

/// A clear, temperate default environment (flir-live's), for fields the
/// caller does not know.
pub fn default_env() -> EnvironmentWire {
    EnvironmentWire {
        ambient_temp_c: 15.0,
        pressure_hpa: 1013.0,
        humidity_pct: f32::NAN,
        dewpoint_c: f32::NAN,
        visibility_m: 20_000.0,
        precip_kind: 0,
        in_cloud: 0,
        _pad: [0; 2],
        precip_rate: 0.0,
        wind_dir_deg: 0.0,
        wind_speed_ms: 0.0,
        lat_deg: 47.6,
        lon_deg: -122.3,
        alt_m: 600.0,
        ground_alt_m: 100.0,
        utc_unix_s: 1_718_928_000.0 + 13.0 * 3600.0,
        sun_dir: [f32::NAN; 3],
    }
}

/// An 8-bit gray picture, row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Gray {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

impl Gray {
    /// Intensity of a sensor image. With the GRAY palette every channel is
    /// the intensity; with a coloured one the brightest channel stands in.
    pub fn from_rgba(img: &RgbaFrame, palette: Palette) -> Gray {
        let data = img
            .rgba
            .chunks_exact(4)
            .map(|p| match palette {
                Palette::Gray => p[0],
                _ => p[0].max(p[1]).max(p[2]),
            })
            .collect();
        Gray { w: img.w, h: img.h, data }
    }

    /// The centre `min(w,h)` square, area-resampled to `size × size`.
    pub fn crop_square(&self, size: usize) -> Gray {
        let side = self.w.min(self.h);
        let size = size.max(1);
        if side == 0 {
            return Gray { w: size, h: size, data: vec![0; size * size] };
        }
        let (x0, y0) = ((self.w - side) / 2, (self.h - side) / 2);
        let scale = side as f32 / size as f32;
        let mut data = Vec::with_capacity(size * size);
        for oy in 0..size {
            let sy0 = y0 + (oy as f32 * scale) as usize;
            let sy1 = (y0 + ((oy + 1) as f32 * scale).ceil() as usize).min(y0 + side).max(sy0 + 1);
            for ox in 0..size {
                let sx0 = x0 + (ox as f32 * scale) as usize;
                let sx1 = (x0 + ((ox + 1) as f32 * scale).ceil() as usize).min(x0 + side).max(sx0 + 1);
                let (mut sum, mut n) = (0u32, 0u32);
                for sy in sy0..sy1.min(self.h) {
                    let row = sy * self.w;
                    for sx in sx0..sx1.min(self.w) {
                        sum += u32::from(self.data[row + sx]);
                        n += 1;
                    }
                }
                data.push((sum + n / 2).checked_div(n).unwrap_or(0) as u8);
            }
        }
        Gray { w: size, h: size, data }
    }
}

/// The persistent develop state. One per stream.
pub struct Pipeline {
    sensor_state: SensorState,
    lut_lwir: RadianceLut,
    lut_mwir: RadianceLut,
    /// The sensor look. Change freely between frames.
    pub params: SensorParams,
    /// The cloud composite's knobs.
    pub clouds: CloudParams,
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new(hud_params())
    }
}

impl Pipeline {
    pub fn new(params: SensorParams) -> Pipeline {
        Pipeline {
            sensor_state: SensorState::default(),
            lut_lwir: RadianceLut::build(Band::Lwir),
            lut_mwir: RadianceLut::build(Band::Mwir),
            params,
            clouds: CloudParams::default(),
        }
    }

    /// Forget AGC history (e.g. after the stream restarts on a new scene).
    pub fn reset(&mut self) {
        self.sensor_state.reset();
    }

    fn lut(&self) -> &RadianceLut {
        if self.params.band_u8 == Band::Mwir as u8 {
            &self.lut_mwir
        } else {
            &self.lut_lwir
        }
    }

    /// Develop one frame. `depth` is the stream's reversed-Z R32F layer,
    /// `w × h`; `rgb` the colour layer as linear RGB (`w × h × 3`), used only
    /// for clouds and skipped when absent or mis-sized.
    pub fn develop(
        &mut self,
        depth: &[f32],
        w: usize,
        h: usize,
        rgb: Option<&[f32]>,
        env: &EnvironmentWire,
    ) -> RgbaFrame {
        let env = Environment::from_wire(env);
        let mut scene = depth_to_scene(depth, w, h, &env, self.lut());
        if let Some(rgb) = rgb {
            clouds::composite(&mut scene, rgb, &env, self.lut(), &self.clouds);
        }
        self.sensor_state.process(&scene, &Sensor::new(self.params))
    }

    /// [`develop`](Self::develop), reduced to the display's square gray
    /// picture.
    pub fn develop_square(
        &mut self,
        depth: &[f32],
        w: usize,
        h: usize,
        rgb: Option<&[f32]>,
        env: &EnvironmentWire,
        size: usize,
    ) -> Gray {
        let img = self.develop(depth, w, h, rgb, env);
        let palette = match self.params.palette_u8 {
            1 => Palette::Gray,
            2 => Palette::Ironbow,
            _ => Palette::PhosphorGreen,
        };
        Gray::from_rgba(&img, palette).crop_square(size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_square_takes_the_centre_and_resamples() {
        // 8x4: left two columns 0, centre 4 columns 200, right two 0.
        let mut data = vec![0u8; 32];
        for y in 0..4 {
            for x in 2..6 {
                data[y * 8 + x] = 200;
            }
        }
        let g = Gray { w: 8, h: 4, data };
        let sq = g.crop_square(2);
        assert_eq!((sq.w, sq.h), (2, 2));
        assert!(sq.data.iter().all(|&v| v == 200), "{:?}", sq.data);
    }

    #[test]
    fn a_sky_and_ground_frame_develops_to_the_requested_square() {
        let (w, h) = (64, 36);
        // Top half sky (0), bottom half ground at ~500 m (0.05 / 500).
        let depth: Vec<f32> = (0..w * h).map(|i| if i / w < h / 2 { 0.0 } else { 1e-4 }).collect();
        let mut p = Pipeline::default();
        let g = p.develop_square(&depth, w, h, None, &default_env(), 32);
        assert_eq!((g.w, g.h, g.data.len()), (32, 32, 32 * 32));
        let top: u32 = g.data[..32 * 4].iter().map(|&v| u32::from(v)).sum();
        let bottom: u32 = g.data[32 * 28..].iter().map(|&v| u32::from(v)).sum();
        assert_ne!(top, bottom, "sky and ground must not develop identically");
    }
}
