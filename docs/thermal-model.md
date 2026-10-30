# Thermal scene model

What each step in `flir-scene/src/shaders/scene.hlsl` computes, and why. The
Rust side (`flir-scene`) supplies the parameter blocks (`SceneConstants`, the
material table, the Planck radiance LUT); the HLSL does the per-pixel maths.
Everything is an engineering approximation tuned for a believable long-wave IR
look, not a radiative-transfer solver; constants live in `material.rs`; the CPU
port in `flir-scene/src/cpu.rs` is what `flir-live` tunes against.

## Per pixel

1. **Geometry.** World position is reconstructed from the depth buffer and the
   inverse view-projection the bridge set for the FLIR view (`worldFromDepth`).
   The depth convention (reversed-Z, infinite far) is determined empirically per
   build (`DepthConvention`, confirmed via the PIX guide) and passed in; an
   unknown convention drops the frame to a lower tier rather than guessing.
   Normals come from the G-buffer in the full tier, else from depth derivatives
   (`normalFromDepth`). The far plane / far depth value is sky.

2. **Material classification.** Soft weights over water, vegetation, bare soil,
   asphalt, concrete, urban, snow/ice, metal and emissive/hot (`classify`). Full
   tier derives them from G-buffer channels (albedo hue, luma, roughness,
   metalness, an HDR-emissive hint, and surface flatness from the normal).
   Depth-only tier guesses from lit colour + flatness. Weights are normalised;
   an empty classification defaults to concrete. The channel assignments in the
   shader are placeholders until `docs/gbuffer-layout.md` is filled from a PIX
   capture.

3. **Surface temperature.**
   `T = T_air(pixel altitude via lapse) + ΔT_class + sun_loading·max(0,N·S)·sunlit`.
   Air temperature falls with altitude from the aircraft's ambient reading at
   6.5 °C/km. Each class adds a steady offset and, by day, sun loading scaled by
   the insolation (`diurnal`) and the surface's `N·S`; by night, retained warmth
   scaled by thermal inertia. So asphalt/concrete stay warm after sunset,
   vegetation cools fast, and water — high inertia, low loading — reads warm at
   night relative to land and cool by day. Precipitation/wetness drags the
   surface toward air temperature and toward water emissivity.

4. **Emission and reflection.**
   `L_surface = ε·B(T) + (1−ε)·L_sky_reflected`, with a Fresnel grazing term so
   smooth water and metal reflect the cold sky at shallow angles (`fres`,
   `grazing`). Non-reflected non-emitted energy is filled with `B(T_air)` so
   low-ε surfaces do not read as holes.

5. **Band radiance `B(T)`.** A lookup of Planck radiance integrated over the
   band (`flir_env::planck`, uploaded as a 1-D texture). LWIR 8–14 µm by
   default; MWIR 3–5 µm optional. The LUT is normalised to 1.0 at 15 °C; MWIR is
   steeper per degree (Wien), giving more contrast — verified in a unit test.

6. **Sky.** Effective clear-sky brightness temperature by elevation
   (`skyTempC`): very cold at zenith, approaching air temperature at the
   horizon, with the zenith depression growing as the column dries.

7. **Clouds.** From the redirected cloud layer: opacity from the premultiplied
   output's alpha, temperature from the cloud base altitude via the lapse rate.
   Composited `L = α·B(T_cloud) + (1−α)·L_behind`, and the atmospheric path is
   shortened to the cloud.

8. **Atmosphere.** `L_observed = τ·L_surface + (1−τ)·B(T_path)` with
   `τ = exp(−k·range)`, `k` from humidity, precipitation and visibility
   (`flir_env::atmos::extinction_per_m`). Distant terrain washes toward the path
   temperature; haze and rain thicken the effect.

## Fallback tiers (`tier.rs`)

Fail closed, automatically, reported in status:

1. **full** — G-buffer + depth + clouds;
2. **depth-only** — geometry, sky, atmosphere, altitude, depth-derivative
   normals, colour-derived material guess;
3. **colour-only** — weighted-RGB heat proxy with a cooling (blue, not
   magenta) sky.

## Deliberate approximations

Per-object temperatures (engines, vehicles, people) and real thermal history are
not modelled; emissive/material cues plus the diurnal curves stand in. **City
lights are not hot in LWIR** — emissive pixels mark warm urban areas or genuine
hot sources, not bright points, which is why the forward-pass billboard lights
must not be treated as heat.
