//! `flir-scene` — the thermal scene model.
//!
//! Turns a captured frame (depth, G-buffer, cloud layer, with the exact
//! view/projection the bridge set) plus an [`flir_env::Environment`] into
//! per-pixel in-band scene radiance, which the sensor model then develops.
//!
//! This crate is **parameter plumbing + HLSL**. The per-pixel maths lives in
//! `shaders/scene.hlsl`; here we define the parameter blocks that drive it, the
//! material table, the depth/geometry conventions, and the fallback tiers. The
//! bridge uploads [`SceneConstants`] and the [`material::MaterialTableGpu`] and
//! dispatches the shader; [`cpu`] is the CPU port `flir-live` runs.
//!
//! See `docs/thermal-model.md` for the physics each term implements.

pub mod cpu;
pub mod material;
pub mod tier;

use infinity_viewports_protocol::Band;

use flir_env::Environment;
use flir_env::planck::RadianceLut;

/// The scene HLSL, embedded so Rust parameter layout and shader stay in lockstep.
pub const HLSL: &str = include_str!("shaders/scene.hlsl");

/// Depth-buffer interpretation, determined empirically per build and recorded
/// in `docs/gbuffer-layout.md`. The shader reconstructs view-space Z from the
/// stored depth using these.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthConvention {
    /// True if the depth buffer is reversed-Z (1 at the near plane).
    pub reversed_z: bool,
    /// True if an infinite far plane is used.
    pub infinite_far: bool,
    pub near_m: f32,
    /// Ignored when `infinite_far`.
    pub far_m: f32,
}

impl Default for DepthConvention {
    fn default() -> Self {
        // MSFS uses reversed-Z with an infinite far plane for the main view;
        // confirmed per build before trusting (DepthConventionUnknown tier drop
        // otherwise). These are the documented defaults, not an assumption the
        // runtime makes silently.
        DepthConvention { reversed_z: true, infinite_far: true, near_m: 0.05, far_m: 1_000.0 }
    }
}

/// Constant-buffer block for the scene shader. `#[repr(C)]`, 16-byte-field
/// aligned to match HLSL cbuffer packing (caller pads as needed).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SceneConstants {
    /// Inverse view-projection the bridge set for the FLIR view, to unproject
    /// depth to world position. Column-major, as HLSL expects.
    pub inv_view_proj: [[f32; 4]; 4],
    /// Camera world position (metres, sim world frame).
    pub cam_world: [f32; 3],
    pub _pad0: f32,
    /// Sun direction (world ENU), from the environment.
    pub sun_dir: [f32; 3],
    pub _pad1: f32,

    /// Air temperature at the aircraft (°C) and lapse rate (°C/m).
    pub ambient_c: f32,
    pub lapse_c_per_m: f32,
    pub aircraft_alt_m: f32,
    pub ground_alt_m: f32,

    /// Diurnal insolation factor [−0.2,1] and a day/night blend [0,1].
    pub diurnal: f32,
    pub is_day: f32,
    /// Atmospheric extinction per metre for the active band.
    pub extinction_per_m: f32,
    /// Precipitation wetness 0..1 (drives surfaces toward air temp + water ε).
    pub wetness: f32,

    /// Water-vapour/path scalar and sky clear-sky depression (°C below air at
    /// zenith).
    pub water_loading: f32,
    pub sky_zenith_depression_c: f32,
    /// Cloud base altitude (m) used when the cloud layer gives no per-pixel
    /// distance.
    pub cloud_base_m: f32,
    /// Active fidelity tier (see `tier::Tier`), 1 full / 2 depth-only / 3 colour.
    pub tier: u32,

    /// Render size.
    pub size: [u32; 2],
    /// Band: 0 LWIR, 1 MWIR (selects which radiance LUT is bound).
    pub band: u32,
    /// Depth convention packed: bit0 reversed_z, bit1 infinite_far.
    pub depth_flags: u32,
    pub depth_near_m: f32,
    pub depth_far_m: f32,
    pub _pad2: [f32; 2],
}

impl SceneConstants {
    /// Build the per-frame constants from an environment, view data and tier.
    #[allow(clippy::too_many_arguments)]
    pub fn build(env: &Environment, band: Band, inv_view_proj: [[f32; 4]; 4], cam_world: [f32; 3], depth: DepthConvention, size: [u32; 2], tier: tier::Tier) -> SceneConstants {
        let depth_flags = (depth.reversed_z as u32) | ((depth.infinite_far as u32) << 1);
        SceneConstants {
            inv_view_proj,
            cam_world,
            _pad0: 0.0,
            sun_dir: env.sun_dir_enu,
            _pad1: 0.0,
            ambient_c: env.ambient_temp_c,
            lapse_c_per_m: flir_env::atmos::LAPSE_RATE_C_PER_M,
            aircraft_alt_m: env.alt_m,
            ground_alt_m: env.ground_alt_m,
            diurnal: env.diurnal_warmth(),
            is_day: if env.is_day() { 1.0 } else { 0.0 },
            extinction_per_m: env.extinction_per_m(band),
            wetness: match env.precip {
                flir_env::Precip::None => 0.0,
                _ => (env.precip_rate / 5.0).clamp(0.0, 1.0),
            },
            water_loading: env.water_loading,
            // Clear-sky zenith is very cold; depression grows as the column dries.
            sky_zenith_depression_c: 40.0 * (1.0 - env.humidity_pct / 100.0).clamp(0.2, 1.0),
            cloud_base_m: env.ground_alt_m + 1000.0,
            tier: tier as u32,
            size,
            band: band as u32,
            depth_flags,
            depth_near_m: depth.near_m,
            depth_far_m: depth.far_m,
            _pad2: [0.0; 2],
        }
    }
}

/// Everything the scene model needs to build one frame's GPU inputs, assembled
/// host/bridge-side.
pub struct SceneModel {
    pub table: [material::MaterialThermal; material::MATERIAL_COUNT],
    pub lwir: RadianceLut,
    pub mwir: RadianceLut,
}

impl SceneModel {
    pub fn new() -> SceneModel {
        SceneModel { table: material::default_table(), lwir: RadianceLut::build(Band::Lwir), mwir: RadianceLut::build(Band::Mwir) }
    }

    pub fn radiance_lut(&self, band: Band) -> &RadianceLut {
        match band {
            Band::Lwir => &self.lwir,
            Band::Mwir => &self.mwir,
        }
    }

    pub fn gpu_table(&self) -> material::MaterialTableGpu {
        material::MaterialTableGpu::from_table(&self.table)
    }
}

impl Default for SceneModel {
    fn default() -> Self {
        Self::new()
    }
}
