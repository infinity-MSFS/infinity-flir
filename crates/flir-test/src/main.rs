use std::time::Instant;

use eframe::egui::{
    self, Color32, ColorImage, ComboBox, Rect, Slider, TextureHandle, TextureOptions, pos2,
};
use infinity_flir::capture::{self, Feed};
use infinity_flir::{
    CHUNK_CHARS, Crop, Encoding, FlirProcessor, FlirSettings, Palette, Polarity, SourceFrame,
    downsample_with,
};

const SETTINGS_KEY: &str = "flir_settings";
const SMOOTH_KEY: &str = "smooth_display";

fn main() -> eframe::Result {
    let args: Vec<String> = std::env::args().collect();
    if let (Some(i), Some(o)) = (arg(&args, "--image"), arg(&args, "--out")) {
        headless(&i, &o);
        return Ok(());
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("FLIR test")
            .with_inner_size([1280.0, 760.0]),
        ..Default::default()
    };
    eframe::run_native(
        "flir-test",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}

/// The MD-10 HUD's symbology green, `display_md_hud::symbology::INK`
/// (70, 255, 110) in 10-systems, as sRGB 0..1.
const MD10_HUD_INK: [f32; 3] = [70.0 / 255.0, 1.0, 110.0 / 255.0];

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn load_still(path: &str, s: &FlirSettings) -> Result<SourceFrame, String> {
    let img = image::open(path).map_err(|e| e.to_string())?.to_rgb32f();
    let (x0, y0, w, h) = s.crop.pixels(img.width() as usize, img.height() as usize);
    Ok(downsample_with(
        w,
        h,
        s.sensor_width,
        |x, y| img.get_pixel((x0 + x) as u32, (y0 + y) as u32).0,
        |c| c,
    ))
}

fn headless(input: &str, output: &str) {
    let mut s = FlirSettings::default();
    if std::env::args().any(|a| a == "--ink-reject") {
        s.ink_reject = Some(MD10_HUD_INK);
    }
    if std::env::args().any(|a| a == "--linear") {
        s.log_domain = false;
        s.equalize = 0.0;
    }
    let num =
        |n: &str| arg(&std::env::args().collect::<Vec<_>>(), n).and_then(|v| v.parse::<f32>().ok());
    if let Some(v) = num("--eq") {
        s.equalize = v;
    }
    if let Some(v) = num("--floor") {
        s.log_floor = v;
    }
    if let Some(v) = num("--plateau") {
        s.plateau = v;
    }
    let hz = &mut s.horizon;
    hz.enabled = !std::env::args().any(|a| a == "--no-horizon");
    hz.show_line = std::env::args().any(|a| a == "--line");
    if let Some(v) = num("--pitch") {
        hz.pitch_deg = v;
    }
    if let Some(v) = num("--bank") {
        hz.bank_deg = v;
    }
    if let Some(v) = num("--vfov") {
        hz.vfov_deg = v;
    }
    if let Some(v) = num("--alt") {
        hz.altitude_ft = v;
    }
    let src = load_still(input, &s).expect("load image");
    let mut p = FlirProcessor::default();
    let mut img = p.process(&src, &s);
    for _ in 0..30 {
        img = p.process(&src, &s);
    }
    image::save_buffer(
        output,
        img.rgba.as_flattened(),
        img.w as u32,
        img.h as u32,
        image::ColorType::Rgba8,
    )
    .expect("save");
    if let Some(path) = arg(&std::env::args().collect::<Vec<_>>(), "--packet") {
        let encoding = if std::env::args().any(|a| a == "--rgba") {
            Encoding::Rgba8
        } else {
            Encoding::Gray8
        };
        std::fs::write(path, img.packet(encoding).to_bytes()).expect("write packet");
    }
}

enum Source {
    None,
    Live(Feed),
    Still { path: String, frame: SourceFrame },
}

struct App {
    settings: FlirSettings,
    smooth_display: bool,
    encoding: Encoding,
    proc: FlirProcessor,
    targets: Vec<(String, capture::Target)>,
    selected: usize,
    source: Source,
    current: Option<SourceFrame>,
    tex: Option<TextureHandle>,
    status: String,
    process_ms: f32,
    packet_stats: (usize, usize),
    geometry: (usize, Crop),
    fps_window: (Instant, u64, u32),
    capture_fps: f32,
    display_fps: f32,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let settings: FlirSettings = cc
            .storage
            .and_then(|s| eframe::get_value(s, SETTINGS_KEY))
            .unwrap_or_default();
        let smooth_display = cc
            .storage
            .and_then(|s| eframe::get_value(s, SMOOTH_KEY))
            .unwrap_or(true);
        let targets = capture::list_targets();
        Self {
            geometry: (settings.sensor_width, settings.crop),
            settings,
            smooth_display,
            encoding: Encoding::Gray8,
            proc: FlirProcessor::default(),
            targets,
            selected: 0,
            source: Source::None,
            current: None,
            tex: None,
            status: if cfg!(windows) {
                "Pick a monitor or the MSFS window and press Start.".into()
            } else {
                "Live capture is Windows-only; load a still image.".into()
            },
            process_ms: 0.0,
            packet_stats: (0, 0),
            fps_window: (Instant::now(), 0, 0),
            capture_fps: 0.0,
            display_fps: 0.0,
        }
    }

    fn start_capture(&mut self, ctx: &egui::Context) {
        self.source = Source::None;
        let Some((name, target)) = self.targets.get(self.selected) else {
            self.status = "No capture target selected.".into();
            return;
        };
        self.proc.reset();
        match Feed::start(target, &self.settings) {
            Ok(feed) => {
                let ctx = ctx.clone();
                feed.on_frame(move || ctx.request_repaint());
                let fmt = if self.settings.capture_hdr {
                    "16-bit float"
                } else {
                    "8-bit"
                };
                self.status = format!("Capturing {name} ({fmt})");
                self.source = Source::Live(feed);
            }
            Err(e) => self.status = e,
        }
    }

    fn load_image(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("image", &["png", "jpg", "jpeg"])
            .pick_file()
        else {
            return;
        };
        let path = path.to_string_lossy().to_string();
        match load_still(&path, &self.settings) {
            Ok(frame) => {
                self.proc.reset();
                self.status = format!("Still: {path}");
                self.source = Source::Still { path, frame };
            }
            Err(e) => self.status = format!("Failed to load image: {e}"),
        }
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        ui.heading("Source");
        let label = self
            .targets
            .get(self.selected)
            .map(|t| t.0.as_str())
            .unwrap_or("(none)");
        ComboBox::from_id_salt("target")
            .width(260.0)
            .selected_text(label)
            .show_ui(ui, |ui| {
                for (i, (name, _)) in self.targets.iter().enumerate() {
                    ui.selectable_value(&mut self.selected, i, name);
                }
            });
        ui.horizontal(|ui| {
            if ui.button("Refresh").clicked() {
                self.targets = capture::list_targets();
                self.selected = self.selected.min(self.targets.len().saturating_sub(1));
            }
            let live = matches!(self.source, Source::Live(_));
            if !live && ui.button("▶ Start").clicked() {
                self.start_capture(&ctx);
            }
            if live && ui.button("■ Stop").clicked() {
                self.source = Source::None;
                self.status = "Stopped.".into();
            }
            if ui.button("Load image…").clicked() {
                self.load_image();
            }
        });
        let live = matches!(self.source, Source::Live(_));
        if ui
            .checkbox(&mut self.settings.capture_hdr, "16-bit float capture")
            .on_hover_text(
                "Captures linear FP16 instead of 8-bit. Most useful with Windows HDR on \
                 and MSFS outputting HDR: shadows keep far more levels for the log stretch.",
            )
            .changed()
            && live
        {
            self.start_capture(&ctx);
        }
        ui.small(&self.status);
        if let Some(f) = &self.current {
            ui.small(format!(
                "sensor {}x{} · process {:.1} ms · capture {:.0} fps · display {:.0} fps",
                f.w, f.h, self.process_ms, self.capture_fps, self.display_fps
            ));
        }
        ui.horizontal(|ui| {
            ui.label("Packet");
            ui.selectable_value(&mut self.encoding, Encoding::Gray8, "Gray8");
            ui.selectable_value(&mut self.encoding, Encoding::Rgba8, "Rgba8");
            let (bytes, chunks) = self.packet_stats;
            ui.small(format!("{:.1} KB · {chunks} chunks", bytes as f32 / 1024.0));
        });

        ui.separator();
        let s = &mut self.settings;
        ui.horizontal(|ui| {
            if ui.button("Defaults").clicked() {
                *s = FlirSettings::default();
            }
            if ui.button("HUD green").clicked() {
                *s = FlirSettings {
                    palette: Palette::Green,
                    polarity: Polarity::WhiteHot,
                    ..s.clone()
                };
            }
            if ui.button("White hot").clicked() {
                *s = FlirSettings {
                    palette: Palette::Gray,
                    polarity: Polarity::WhiteHot,
                    ..s.clone()
                };
            }
            if ui.button("Black hot").clicked() {
                *s = FlirSettings {
                    palette: Palette::Gray,
                    polarity: Polarity::BlackHot,
                    ..s.clone()
                };
            }
        });
        ui.checkbox(&mut s.show_source, "Bypass (show source)");

        egui::CollapsingHeader::new("Sensor")
            .default_open(true)
            .show(ui, |ui| {
                ui.add(
                    Slider::new(&mut s.sensor_width, 160..=1280)
                        .step_by(16.0)
                        .text("width px"),
                );
                ui.add(Slider::new(&mut s.persistence, 0.0..=0.95).text("persistence"));
                ui.checkbox(&mut self.smooth_display, "Smooth display filtering");
            });

        egui::CollapsingHeader::new("Crop")
            .default_open(true)
            .show(ui, |ui| {
                let c = &mut s.crop;
                ui.add(Slider::new(&mut c.x, 0.0..=0.95).text("left"));
                ui.add(Slider::new(&mut c.y, 0.0..=0.95).text("top"));
                ui.add(Slider::new(&mut c.w, 0.05..=1.0).text("width"));
                ui.add(Slider::new(&mut c.h, 0.05..=1.0).text("height"));
                if ui.button("Reset crop").clicked() {
                    *c = Crop::default();
                }
            });

        egui::CollapsingHeader::new("HUD ink reject")
            .default_open(true)
            .show(ui, |ui| {
                let mut on = s.ink_reject.is_some();
                if ui.checkbox(&mut on, "Remove the HUD's own light").changed() {
                    s.ink_reject = on.then_some(MD10_HUD_INK);
                }
                if let Some(ink) = s.ink_reject.as_mut() {
                    ui.horizontal(|ui| {
                        ui.color_edit_button_rgb(ink);
                        ui.label("ink");
                    });
                }
                ui.add(Slider::new(&mut s.ink_mask, 0.0..=1.0).text("stroke refill above"));
            });

        egui::CollapsingHeader::new("Heat proxy")
            .default_open(true)
            .show(ui, |ui| {
                ui.add(Slider::new(&mut s.weights[0], -1.0..=1.5).text("red weight"));
                ui.add(Slider::new(&mut s.weights[1], -1.0..=1.5).text("green weight"));
                ui.add(Slider::new(&mut s.weights[2], -1.0..=1.5).text("blue weight"));
                ui.add(Slider::new(&mut s.sky_suppression, 0.0..=4.0).text("sky → cold"));
            });

        egui::CollapsingHeader::new("Gain / AGC")
            .default_open(true)
            .show(ui, |ui| {
                ui.checkbox(&mut s.log_domain, "Log intensity (lift darks)");
                ui.add_enabled(
                    s.log_domain,
                    Slider::new(&mut s.log_floor, 0.001..=0.2)
                        .logarithmic(true)
                        .text("log floor"),
                );
                ui.add(Slider::new(&mut s.equalize, 0.0..=1.0).text("equalization"));
                ui.add_enabled(
                    s.equalize > 0.0,
                    Slider::new(&mut s.plateau, 0.2..=20.0)
                        .logarithmic(true)
                        .text("plateau"),
                );
                ui.checkbox(&mut s.agc, "Auto gain (histogram)");
                ui.add_enabled(
                    s.agc,
                    Slider::new(&mut s.agc_low_pct, 0.0..=20.0).text("low clip %"),
                );
                ui.add_enabled(
                    s.agc,
                    Slider::new(&mut s.agc_high_pct, 80.0..=100.0).text("high clip %"),
                );
                ui.add_enabled(
                    s.agc,
                    Slider::new(&mut s.agc_speed, 0.01..=1.0)
                        .logarithmic(true)
                        .text("AGC speed"),
                );
                ui.add(Slider::new(&mut s.gain, 0.2..=4.0).text("gain"));
                ui.add(Slider::new(&mut s.level, -0.5..=0.5).text("level"));
                ui.add(Slider::new(&mut s.gamma, 0.3..=3.0).text("gamma"));
            });

        egui::CollapsingHeader::new("Horizon model")
            .default_open(true)
            .show(ui, |ui| {
                let hz = &mut s.horizon;
                ui.checkbox(&mut hz.enabled, "Attitude-driven background");
                ui.add_enabled_ui(hz.enabled, |ui| {
                    ui.small("Manual for now; the real app reads these from SimConnect.");
                    ui.add(Slider::new(&mut hz.pitch_deg, -30.0..=30.0).text("pitch °"));
                    ui.add(Slider::new(&mut hz.bank_deg, -60.0..=60.0).text("bank °"));
                    ui.add(Slider::new(&mut hz.vfov_deg, 5.0..=100.0).text("vertical FOV °"));
                    ui.add(
                        Slider::new(&mut hz.altitude_ft, 10.0..=45_000.0)
                            .logarithmic(true)
                            .text("altitude ft"),
                    );
                    ui.checkbox(&mut hz.show_line, "Show horizon line");
                    ui.add(Slider::new(&mut hz.ground_temp, 0.0..=1.0).text("ground temp"));
                    ui.add(Slider::new(&mut hz.horizon_temp, 0.0..=1.0).text("horizon temp"));
                    ui.add(Slider::new(&mut hz.zenith_temp, 0.0..=1.0).text("zenith temp"));
                    ui.add(Slider::new(&mut hz.sky_falloff_deg, 1.0..=45.0).text("sky falloff °"));
                    ui.add(
                        Slider::new(&mut hz.atten_km, 5.0..=300.0)
                            .logarithmic(true)
                            .text("haze range km"),
                    );
                    ui.add(Slider::new(&mut hz.ground_vis, 0.0..=2.0).text("image weight ground"));
                    ui.add(Slider::new(&mut hz.sky_vis, 0.0..=2.0).text("image weight sky"));
                });
            });

        egui::CollapsingHeader::new("Detail")
            .default_open(true)
            .show(ui, |ui| {
                ui.add(Slider::new(&mut s.detail, 0.0..=6.0).text("detail boost"));
                ui.add(Slider::new(&mut s.detail_radius, 1..=40).text("radius px"));
            });

        egui::CollapsingHeader::new("Look")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut s.polarity, Polarity::WhiteHot, "White hot");
                    ui.selectable_value(&mut s.polarity, Polarity::BlackHot, "Black hot");
                });
                ui.horizontal(|ui| {
                    for (p, n) in [
                        (Palette::Gray, "Gray"),
                        (Palette::Green, "Green"),
                        (Palette::Amber, "Amber"),
                        (Palette::Ironbow, "Ironbow"),
                    ] {
                        ui.selectable_value(&mut s.palette, p, n);
                    }
                });
                ui.add(Slider::new(&mut s.noise, 0.0..=0.2).text("noise"));
                ui.add(Slider::new(&mut s.fpn, 0.0..=0.08).text("column FPN"));
                ui.add(Slider::new(&mut s.vignette, 0.0..=1.0).text("vignette"));
            });
    }

    fn update_frame(&mut self, ctx: &egui::Context) {
        let geometry = (self.settings.sensor_width, self.settings.crop);
        if geometry != self.geometry {
            self.geometry = geometry;
            match &mut self.source {
                Source::Live(feed) => feed.configure(&self.settings),
                Source::Still { path, frame } => {
                    if let Ok(f) = load_still(path, &self.settings) {
                        *frame = f;
                    }
                }
                Source::None => {}
            }
        }

        let fresh = match &self.source {
            Source::Live(feed) => feed.latest(),
            _ => None,
        };
        if let Some(f) = fresh {
            self.current = Some(f);
        }
        let src = match &self.source {
            Source::Still { frame, .. } => Some(frame),
            _ => self.current.as_ref(),
        };
        let Some(src) = src else { return };

        let t = Instant::now();
        let img = self.proc.process(src, &self.settings);
        self.process_ms = self.process_ms * 0.9 + t.elapsed().as_secs_f32() * 1000.0 * 0.1;

        let bytes = img.packet(self.encoding).to_bytes().len();
        self.packet_stats = (bytes, (bytes.div_ceil(3) * 4).div_ceil(CHUNK_CHARS));
        let img = ColorImage::from_rgba_unmultiplied([img.w, img.h], img.rgba.as_flattened());
        let opts = if self.smooth_display {
            TextureOptions::LINEAR
        } else {
            TextureOptions::NEAREST
        };
        match &mut self.tex {
            Some(tex) => tex.set(img, opts),
            None => self.tex = Some(ctx.load_texture("flir", img, opts)),
        }

        if matches!(self.source, Source::Still { .. }) {
            ctx.request_repaint();
        }

        let (start, frames0, shown) = &mut self.fps_window;
        *shown += 1;
        let dt = start.elapsed().as_secs_f32();
        if dt >= 1.0 {
            let frames = match &self.source {
                Source::Live(feed) => feed.frames(),
                _ => *frames0,
            };
            self.capture_fps = (frames - *frames0) as f32 / dt;
            self.display_fps = *shown as f32 / dt;
            *start = Instant::now();
            *frames0 = frames;
            *shown = 0;
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.update_frame(&ctx);

        egui::Panel::left("controls")
            .resizable(true)
            .default_size(320.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui));
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(Color32::BLACK))
            .show(ui, |ui| {
                let avail = ui.max_rect();
                let Some(tex) = &self.tex else {
                    ui.centered_and_justified(|ui| ui.label("No source"));
                    return;
                };
                let [tw, th] = tex.size();
                let scale = (avail.width() / tw as f32).min(avail.height() / th as f32);
                let size = egui::vec2(tw as f32 * scale, th as f32 * scale);
                let rect = Rect::from_center_size(avail.center(), size);
                ui.painter().image(
                    tex.id(),
                    rect,
                    Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                    Color32::WHITE,
                );
            });
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, SETTINGS_KEY, &self.settings);
        eframe::set_value(storage, SMOOTH_KEY, &self.smooth_display);
    }
}
