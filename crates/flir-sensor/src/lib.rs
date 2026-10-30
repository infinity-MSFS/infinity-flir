//! `flir-sensor` — the sensor model that turns scene band-radiance into the
//! displayed FLIR image.
//!
//! Stages (ported from the CPU prototype in `infinity-flir`, tunable in
//! `flir-live`):
//!
//! 1. **AGC** — percentile stretch blended with plateau histogram
//!    equalization; the stretch limits are smoothed frame-to-frame.
//! 2. **Base/detail split** — box-blur low pass; detail scaled by the mapping
//!    slope and added back (local-contrast / DDE).
//! 3. **Gain / level / gamma**, then **polarity** (white-hot / black-hot).
//! 4. **Noise** — temporal shot noise, column fixed-pattern noise, optional
//!    non-uniformity drift; slight **halo** at high-contrast edges; **vignette**.
//! 5. **Palette** — phosphor green (default), gray, ironbow.
//!
//! This crate is the authoritative CPU reference; the real-time path runs the
//! equivalent [`HLSL`] in the bridge. Keeping both here lets the shader be
//! validated against the reference on the same frame.

pub mod cpu;

use infinity_viewports_protocol::{Palette, Polarity, SensorParams};

/// The sensor HLSL, for the future GPU path. Embedded next to the CPU
/// reference so the two never drift.
pub const HLSL: &str = include_str!("shaders/sensor.hlsl");

/// Resolved sensor configuration (decoded from the wire `SensorParams`).
#[derive(Clone, Copy, Debug)]
pub struct Sensor {
    pub params: SensorParams,
}

impl Sensor {
    pub fn new(params: SensorParams) -> Sensor {
        Sensor { params }
    }
    pub fn polarity(&self) -> Polarity {
        if self.params.polarity_u8 == 1 { Polarity::BlackHot } else { Polarity::WhiteHot }
    }
    pub fn palette(&self) -> Palette {
        match self.params.palette_u8 {
            1 => Palette::Gray,
            2 => Palette::Ironbow,
            _ => Palette::PhosphorGreen,
        }
    }
}

/// Map a normalised intensity `v` in [0,1] to display RGB for a palette. Shared
/// by the CPU path and documented so the HLSL matches exactly.
pub fn palette_rgb(p: Palette, v: f32) -> [f32; 3] {
    let v = v.clamp(0.0, 1.0);
    match p {
        Palette::Gray => [v, v, v],
        // Phosphor green: green-dominant with a faint desaturating tail.
        Palette::PhosphorGreen => [v * v * 0.45, v, v * v * 0.35],
        Palette::Ironbow => ironbow(v),
    }
}

fn ironbow(v: f32) -> [f32; 3] {
    const STOPS: [(f32, [f32; 3]); 6] = [
        (0.00, [0.00, 0.00, 0.00]),
        (0.20, [0.13, 0.00, 0.45]),
        (0.45, [0.70, 0.05, 0.55]),
        (0.70, [0.98, 0.45, 0.05]),
        (0.90, [1.00, 0.85, 0.20]),
        (1.00, [1.00, 1.00, 0.95]),
    ];
    for pair in STOPS.windows(2) {
        let (a, ca) = pair[0];
        let (b, cb) = pair[1];
        if v <= b {
            let t = ((v - a) / (b - a)).clamp(0.0, 1.0);
            return [ca[0] + (cb[0] - ca[0]) * t, ca[1] + (cb[1] - ca[1]) * t, ca[2] + (cb[2] - ca[2]) * t];
        }
    }
    STOPS[5].1
}
