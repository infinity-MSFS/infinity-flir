mod capture;
mod flir;
mod horizon;

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use eframe::egui::{self, Color32, ComboBox, Rect, Slider, TextureHandle, TextureOptions, pos2};
use flir::{FlirProcessor, FlirSettings, Palette, Polarity, SourceFrame, downsample_with};

const SETTINGS_KEY: &str = "flir_settings";

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

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn load_still(path: &str, sensor_w: usize) -> Result<SourceFrame, String> {
    let img = image::open(path).map_err(|e| e.to_string())?.to_rgb32f();
    let (w, h) = (img.width() as usize, img.height() as usize);
    Ok(downsample_with(
        w,
        h,
        sensor_w,
        |x, y| img.get_pixel(x as u32, y as u32).0,
        |c| c,
    ))
}

fn headless(input: &str, output: &str) {
    let mut s = FlirSettings::default();
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
    let src = load_still(input, s.sensor_width).expect("load image");
    let mut p = FlirProcessor::default();
    let mut img = p.process(&src, &s);
    for _ in 0..30 {
        img = p.process(&src, &s);
    }
    let bytes: Vec<u8> = img
        .pixels
        .iter()
        .flat_map(|c| [c.r(), c.g(), c.b()])
        .collect();
    image::save_buffer(
        output,
        &bytes,
        img.size[0] as u32,
        img.size[1] as u32,
        image::ColorType::Rgb8,
    )
    .expect("save");
}

enum Source {
    None,
    Live(capture::CaptureHandle),
    Still { path: String, frame: SourceFrame },
}

struct App {
    settings: FlirSettings,
    proc: FlirProcessor,
    shared: Arc<capture::Shared>,
    targets: Vec<(String, capture::Target)>,
    selected: usize,
    source: Source,
    current: Option<SourceFrame>,
    tex: Option<TextureHandle>,
    status: String,
    process_ms: f32,
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
        let shared = capture::Shared::new(settings.sensor_width);
        let targets = capture::list_targets();
        Self {
            settings,
            proc: FlirProcessor::default(),
            shared,
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
        let hdr = self.settings.capture_hdr;
        match capture::start(target, self.shared.clone(), ctx.clone(), hdr) {
            Ok(h) => {
                let fmt = if hdr { "16-bit float" } else { "8-bit" };
                self.status = format!("Capturing {name} ({fmt})");
                self.source = Source::Live(h);
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
        match load_still(&path, self.settings.sensor_width) {
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
                ui.checkbox(&mut s.smooth_display, "Smooth display filtering");
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
        if self
            .shared
            .sensor_w
            .swap(self.settings.sensor_width, Ordering::Relaxed)
            != self.settings.sensor_width
        {
            if let Source::Still { path, frame } = &mut self.source {
                if let Ok(f) = load_still(path, self.settings.sensor_width) {
                    *frame = f;
                }
            }
        }

        let fresh = match &self.source {
            Source::Live(_) => self.shared.latest.lock().unwrap().take(),
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

        let opts = if self.settings.smooth_display {
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
            let frames = self.shared.frames.load(Ordering::Relaxed);
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
    }
}
