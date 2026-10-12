# Infinity FLIR

An attempt to add flir to MSFS. we will extract each frame, apply a convincing enough FLIR filter essentially and stream back into the gauges over the wasm bridge

| Crate | Role |
|-------|------|
| `infinity-flir` | lib: window capture → crop → FLIR filter → raw packet → bridge chunks |
| `flir-test` | bin: egui tuning GUI, and headless `--image in.png --out out.png [--packet out.bin]` |

## Usage

Host app (manager):

```rust
use infinity_flir::capture::{self, Feed};
use infinity_flir::{CHUNK_CHARS, Encoding, FlirProcessor, FlirSettings, chunks};

let mut s = FlirSettings::default();
let target = capture::find_window("Microsoft Flight Simulator").unwrap();
let feed = Feed::start(&target, &s)?;
let mut flir = FlirProcessor::default();

for seq in 0.. {
    let Some(src) = feed.wait(Duration::from_millis(100)) else { continue };
    s.horizon.pitch_deg = pitch;
    s.horizon.bank_deg = bank;
    s.horizon.altitude_ft = alt;
    let packet = flir.process(&src, &s).packet(Encoding::Gray8);
    for chunk in chunks(seq, &packet, CHUNK_CHARS) {
        mgr.command(ns, "flir.chunk", serde_json::to_value(chunk)?, timeout).await?;
    }
}
```

Gauge (`infinity-flir = { default-features = false }`):

```rust
let mut rx = Reassembler::default();

if let Some(Ok(packet)) = rx.push(serde_json::from_value(payload)?) {
    nvg.update_image(image, &packet.to_rgba());
}
```

Packet layout: `"FL"`, encoding (`0` Gray8, `1` Rgba8), pad, width u16 LE, height u16 LE, pixels.

## Looking through a HUD

A host filming the sim window through a collimated HUD sets two things per frame:

- `FlirSettings::warp` — a `Warp` (3x3 homography, output unit square -> viewport unit square) that resamples only the region behind the HUD onto the HUD's own pixel grid (`downsample_warped`). `crop` then names the sim's render viewport inside the capture. Off-screen and behind-the-eye samples read black. `Feed::source_size` gives the capture size the warp is built for.
- `FlirSettings::ink_reject` — the HUD's ink colour (sRGB 0..1). The HUD only ADDS that colour, so `reject_ink` measures brightness along a colour axis with no ink in it and removes the HUD's own light (symbology and the previous FLIR frame) before filtering; clipped strokes above `ink_mask` are refilled from their neighbours. `flir-test --ink-reject` (or the GUI's HUD ink reject panel) exercises it with the MD-10's green.
