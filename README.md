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
