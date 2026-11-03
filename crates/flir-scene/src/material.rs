//! Material classes and their thermal properties.
//!
//! The classifier produces *soft weights* over these classes from G-buffer
//! channels (albedo, roughness/metalness, emissive, material flags). The table
//! here is data-driven and tunable: each class carries the
//! physical constants the surface-temperature and emission model reads.

use serde::{Deserialize, Serialize};

/// The material classes, in a fixed order that matches the weight vector the
/// classifier emits and the `MATERIAL_COUNT` arrays in the HLSL.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterialClass {
    Water = 0,
    Vegetation = 1,
    BareSoil = 2,
    Asphalt = 3,
    Concrete = 4,
    Urban = 5,
    SnowIce = 6,
    Metal = 7,
    /// Emissive / hot source (engines, fires, warm windows).
    Emissive = 8,
}

pub const MATERIAL_COUNT: usize = 9;

/// Thermal constants for one material class.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct MaterialThermal {
    /// Broadband IR emissivity at normal incidence (0..1).
    pub emissivity: f32,
    /// Steady-state offset from local air temperature at thermal equilibrium
    /// with no sun, °C. Water ~0 (inert, tracks a slow mean); tarmac slightly
    /// positive from stored heat; vegetation slightly negative.
    pub base_offset_c: f32,
    /// Sun-loading coefficient: peak °C added per unit insolation·(N·S) when
    /// fully sunlit. Asphalt/concrete high, water ~0, snow low (high albedo).
    pub sun_loading_c: f32,
    /// Thermal inertia 0..1: how strongly the surface resists diurnal swing and
    /// retains heat after sunset. Water ~1 (inert/warm at night); vegetation ~0
    /// (cools fast); concrete/asphalt high.
    pub inertia: f32,
    /// How strongly precipitation/wetness drags the surface toward air temp and
    /// toward water emissivity (0..1).
    pub wetting: f32,
    /// Fresnel strength for sky reflection: 1 for smooth water/metal (reflects
    /// cold sky at grazing angles), ~0 for diffuse land.
    pub fresnel: f32,
}

impl MaterialThermal {
    pub const fn new(
        emissivity: f32,
        base_offset_c: f32,
        sun_loading_c: f32,
        inertia: f32,
        wetting: f32,
        fresnel: f32,
    ) -> Self {
        Self {
            emissivity,
            base_offset_c,
            sun_loading_c,
            inertia,
            wetting,
            fresnel,
        }
    }
}

/// Default thermal table. These are physically-motivated starting points, not
/// measured constants; tune them against reference imagery.
pub fn default_table() -> [MaterialThermal; MATERIAL_COUNT] {
    use MaterialClass::*;
    let mut t = [MaterialThermal::new(0.95, 0.0, 0.0, 0.5, 0.0, 0.0); MATERIAL_COUNT];
    //                                        ε     base  sun  inertia wet  fresnel
    t[Water as usize] = MaterialThermal::new(0.98, 1.0, 0.5, 0.95, 0.0, 1.0);
    t[Vegetation as usize] = MaterialThermal::new(0.97, -1.0, 3.0, 0.15, 0.3, 0.0);
    t[BareSoil as usize] = MaterialThermal::new(0.93, 0.5, 6.0, 0.35, 0.5, 0.05);
    t[Asphalt as usize] = MaterialThermal::new(0.95, 2.0, 12.0, 0.8, 0.2, 0.1);
    t[Concrete as usize] = MaterialThermal::new(0.92, 1.5, 9.0, 0.75, 0.2, 0.1);
    t[Urban as usize] = MaterialThermal::new(0.90, 2.5, 8.0, 0.7, 0.1, 0.15);
    t[SnowIce as usize] = MaterialThermal::new(0.99, -2.0, 1.0, 0.6, 0.1, 0.2);
    t[Metal as usize] = MaterialThermal::new(0.25, 0.0, 10.0, 0.3, 0.0, 1.0);
    t[Emissive as usize] = MaterialThermal::new(0.95, 40.0, 0.0, 0.2, 0.0, 0.0);
    t
}

/// Flattened GPU form of the thermal table (one float array per field), ready
/// to drop into a constant buffer matching `scene.hlsl`.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct MaterialTableGpu {
    pub emissivity: [f32; MATERIAL_COUNT],
    pub base_offset_c: [f32; MATERIAL_COUNT],
    pub sun_loading_c: [f32; MATERIAL_COUNT],
    pub inertia: [f32; MATERIAL_COUNT],
    pub wetting: [f32; MATERIAL_COUNT],
    pub fresnel: [f32; MATERIAL_COUNT],
}

impl MaterialTableGpu {
    pub fn from_table(t: &[MaterialThermal; MATERIAL_COUNT]) -> Self {
        let mut g = MaterialTableGpu {
            emissivity: [0.0; MATERIAL_COUNT],
            base_offset_c: [0.0; MATERIAL_COUNT],
            sun_loading_c: [0.0; MATERIAL_COUNT],
            inertia: [0.0; MATERIAL_COUNT],
            wetting: [0.0; MATERIAL_COUNT],
            fresnel: [0.0; MATERIAL_COUNT],
        };
        #[allow(clippy::needless_range_loop)]
        for i in 0..MATERIAL_COUNT {
            g.emissivity[i] = t[i].emissivity;
            g.base_offset_c[i] = t[i].base_offset_c;
            g.sun_loading_c[i] = t[i].sun_loading_c;
            g.inertia[i] = t[i].inertia;
            g.wetting[i] = t[i].wetting;
            g.fresnel[i] = t[i].fresnel;
        }
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn water_is_inert_and_reflective() {
        let t = default_table();
        let w = t[MaterialClass::Water as usize];
        assert!(w.inertia > 0.9);
        assert!(w.fresnel > 0.9);
        assert!(w.sun_loading_c < 1.0);
    }

    #[test]
    fn asphalt_retains_more_than_vegetation() {
        let t = default_table();
        assert!(
            t[MaterialClass::Asphalt as usize].inertia
                > t[MaterialClass::Vegetation as usize].inertia
        );
        assert!(
            t[MaterialClass::Asphalt as usize].sun_loading_c
                > t[MaterialClass::Vegetation as usize].sun_loading_c
        );
    }
}
