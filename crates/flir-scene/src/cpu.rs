//! CPU scene-radiance proxies: turn a captured depth (and optional normals)
//! frame plus an [`Environment`] into a single-channel in-band radiance frame
//! for the sensor model. This is the honest subset of the full `scene.hlsl`
//! model that depth/geometry alone support (cold sky, warm ground, range
//! extinction, geometry relief); the full model additionally needs materials
//! and the camera matrices.
//!
//! `flir-live` develops the sim's streamed buffers through this.

use flir_env::Environment;
use flir_env::planck::RadianceLut;
use flir_sensor::cpu::ScalarFrame;
use flir_sensor::par;
use flir_types::Band;

/// Reversed-Z near plane used to reconstruct range as `near / depth`.
const NEAR_M: f32 = 0.05;

/// Depth-only thermal scene proxy → scalar in-band radiance per pixel.
///
/// * sky (depth ≈ 0, reversed-Z infinite far) → cold, colder toward zenith;
/// * ground → air temperature at ground level plus diurnal surface warmth,
///   modulated by a depth-derivative relief term;
/// * atmosphere: `L = τ·L_surface + (1−τ)·B(T_path)`, `τ = exp(−k·range)`,
///   `range = near / depth`.
pub fn depth_to_scene(
    depth: &[f32],
    w: usize,
    h: usize,
    env: &Environment,
    lut: &RadianceLut,
) -> ScalarFrame {
    let ext = env.extinction_per_m(Band::Lwir);
    let ambient = env.ambient_temp_c;
    let day = env.is_day();
    let diurnal = env.diurnal_warmth().max(0.0);
    let t_ground_air = env.air_temp_at(env.ground_alt_m);
    let t_path = env.air_temp_at((env.alt_m + env.ground_alt_m) * 0.5);
    let surface_warm = if day { 12.0 * diurnal } else { 3.0 };
    let sky_depr = 40.0 * (1.0 - env.humidity_pct / 100.0).clamp(0.2, 1.0);

    let samp = |x: isize, y: isize| {
        let xi = x.clamp(0, w as isize - 1) as usize;
        let yi = y.clamp(0, h as isize - 1) as usize;
        depth[yi * w + xi]
    };
    let l_path = lut.lookup(t_path);
    let mut data = vec![0f32; w * h];
    par::rows(&mut data, w, |y, row| {
        for (x, out) in row.iter_mut().enumerate() {
            let d = depth[y * w + x];
            if d <= 1e-6 {
                let elev = 1.0 - (y as f32 / h as f32);
                *out = lut.lookup(ambient - sky_depr * elev);
                continue;
            }
            let range = (NEAR_M / d).max(NEAR_M);
            // Relief from depth *curvature*, not slope. Under perspective the
            // reversed-Z depth of any planar surface is linear in screen space,
            // so its second difference (Laplacian) is ~0 for flat ground at ANY
            // viewing angle. The old first-difference slope term instead
            // saturated at grazing angles — turning flat terrain near the
            // horizon into a field of black-cracked facets. The Laplacian
            // responds only to real bumps and depth edges; normalising by depth
            // keeps its scale comparable near and far. A sky neighbour (d≈0) is
            // treated as flat so the horizon line itself is not drawn as relief.
            let (xi, yi) = (x as isize, y as isize);
            let nb = |v: f32| if v <= 1e-6 { d } else { v };
            let lap_x = nb(samp(xi + 1, yi)) + nb(samp(xi - 1, yi)) - 2.0 * d;
            let lap_y = nb(samp(xi, yi + 1)) + nb(samp(xi, yi - 1)) - 2.0 * d;
            let relief = ((lap_x.abs() + lap_y.abs()) / d.max(1e-5) * 6.0).clamp(0.0, 1.0);
            let t_surf = t_ground_air + surface_warm - relief * 2.5;
            let l_surf = lut.lookup(t_surf);
            let tau = (-ext * range).exp();
            *out = tau * l_surf + (1.0 - tau) * l_path;
        }
    });
    ScalarFrame { w, h, data }
}

/// Full-geometry scene: the depth-only proxy, but with the depth-gradient relief
/// replaced by real surface normals when present. Normals drive a Lambert
/// thermal-shading term (sun-facing slopes warmer, shadowed cooler); sky, range
/// and atmosphere are identical to the depth-only path. Materials and the camera
/// matrices are still pending, so this is "geometry-full", not material-full.
pub fn gbuffer_to_scene(
    depth: &[f32],
    normal: Option<&[[f32; 3]]>,
    w: usize,
    h: usize,
    env: &Environment,
    lut: &RadianceLut,
) -> ScalarFrame {
    // Without this frame's camera matrices we cannot put normals or the sun into
    // a shared world frame, so we shade with a fixed light in the normal's own
    // space — this reveals true surface relief regardless of the encoding frame.
    let light = {
        let l = [0.30f32, 0.75, 0.58];
        let m = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt();
        [l[0] / m, l[1] / m, l[2] / m]
    };

    let ext = env.extinction_per_m(Band::Lwir);
    let ambient = env.ambient_temp_c;
    let day = env.is_day();
    let diurnal = env.diurnal_warmth().max(0.0);
    let t_ground_air = env.air_temp_at(env.ground_alt_m);
    let t_path = env.air_temp_at((env.alt_m + env.ground_alt_m) * 0.5);
    let surface_warm = if day { 8.0 * diurnal } else { 3.0 };
    let sun_amp = if day { 7.0 * diurnal.max(0.25) } else { 2.0 };
    let sky_depr = 40.0 * (1.0 - env.humidity_pct / 100.0).clamp(0.2, 1.0);

    let samp = |x: isize, y: isize| {
        let xi = x.clamp(0, w as isize - 1) as usize;
        let yi = y.clamp(0, h as isize - 1) as usize;
        depth[yi * w + xi]
    };
    let mut data = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let d = depth[idx];
            if d <= 1e-6 {
                let elev = 1.0 - (y as f32 / h as f32);
                data[idx] = lut.lookup(ambient - sky_depr * elev);
                continue;
            }
            let range = (NEAR_M / d).max(NEAR_M);
            let shade = if let Some(n) = normal {
                let nn = n[idx];
                let ndl = (nn[0] * light[0] + nn[1] * light[1] + nn[2] * light[2]).clamp(-1.0, 1.0);
                (ndl - 0.1) * sun_amp
            } else {
                // Depth-curvature relief (planarity-exact under perspective); see
                // `depth_to_scene`. Avoids the grazing-angle faceting the old
                // slope term produced on flat terrain.
                let (xi, yi) = (x as isize, y as isize);
                let nb = |v: f32| if v <= 1e-6 { d } else { v };
                let lap_x = nb(samp(xi + 1, yi)) + nb(samp(xi - 1, yi)) - 2.0 * d;
                let lap_y = nb(samp(xi, yi + 1)) + nb(samp(xi, yi - 1)) - 2.0 * d;
                -((lap_x.abs() + lap_y.abs()) / d.max(1e-5) * 6.0).clamp(0.0, 1.0) * 2.5
            };
            let t_surf = t_ground_air + surface_warm + shade;
            let l_surf = lut.lookup(t_surf);
            let tau = (-ext * range).exp();
            data[idx] = tau * l_surf + (1.0 - tau) * lut.lookup(t_path);
        }
    }
    ScalarFrame { w, h, data }
}

/// Decode a signed-octahedral normal (two [0,1] channels) to a unit vector. The
/// exact engine encoding is unconfirmed; this is the standard scheme.
pub fn oct_decode(e0: f32, e1: f32) -> [f32; 3] {
    let (fx, fy) = (e0 * 2.0 - 1.0, e1 * 2.0 - 1.0);
    let mut n = [fx, fy, 1.0 - fx.abs() - fy.abs()];
    let t = (-n[2]).max(0.0);
    n[0] += if n[0] >= 0.0 { -t } else { t };
    n[1] += if n[1] >= 0.0 { -t } else { t };
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-6);
    [n[0] / len, n[1] / len, n[2] / len]
}
