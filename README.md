# Infinity FLIR

Physically-based long-wave IR (FLIR / EVS) for MSFS 2024. Consumes the extra
nose viewport from [infinity-viewports](https://github.com/infinity-MSFS/infinity-viewports)
(depth + lit colour) and develops a believable 8–14 µm image from it. It is
**not** a recolour of a visible image.

## Crates

| Crate         | Kind       | Responsibility                                                                 |
| ------------- | ---------- | ------------------------------------------------------------------------------ |
| `flir-env`    | lib        | Environment and derived values: NOAA sun, lapse, humidity, extinction, Planck. |
| `flir-scene`  | lib + HLSL | Thermal scene model; `cpu.rs` is the depth-driven port used live.              |
| `flir-sensor` | lib + HLSL | Sensor model: AGC, detail, noise, FPN, halo, polarity, palette.                |
| `flir-live`   | dev bin    | egui tuner: colour vs thermal side by side, live sensor/env/camera sliders.    |
| `flir-pipeline` | lib      | Headless develop for a production consumer: layer decode, cloud composite, scene + sensor, crop/resample to a square display. |
| `flir-stream` | lib (wasm) | The picture's wire: gauge state report (with environment), delta+deflate codec, chunks, reassembly. Shared by the manager and the in-sim gauge. |

## Production path

The Infinity Manager (`infinity-manager-v3`, `crates/flir` + `flir_handler`)
attaches with infinity-viewports, mounts the nose camera on the airframe,
develops each frame with `flir-pipeline` and streams it with `flir-stream` to
the MD-10 HUD gauge (`10-systems`, `display-md-hud`), which builds against
`flir-stream` too.

## Building

```
cargo build --workspace
cargo test  --workspace
```

## Running

Build the bridge DLL in infinity-viewports first (with MSFS closed), then:

```
cargo run -p flir-live -- --bridge ..\infinity-viewports\target\debug\infinity_viewports_bridge.dll
```

See `docs/thermal-model.md` and `docs/environment-simvars.md`.
