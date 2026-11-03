//! `flir-live` — live FLIR tuner.
//!
//! Attaches to the running sim through `infinity-viewports-host` exactly as the Manager would,
//! starts the nose-viewport video stream (colour + depth, 1024×576), and opens
//! the shared video mapping. Each frame it decodes the streamed colour buffer
//! for a reference view and develops the thermal image on the CPU from the
//! streamed depth, through the `flir-scene` proxy + `flir-sensor` pipeline —
//! so every sensor and environment knob can be
//! tuned live against the sim's real buffers, colour and thermal side by side.
//!
//! Usage:
//!   flir-live --bridge <bridge.dll>
//!
//! The thermal develop runs entirely in this process. Sensor and environment
//! sliders never touch the bridge. The camera sliders write the nose driver's
//! camera block, and `stream:start` plus the colour-source selector talk to the bridge.

use std::path::PathBuf;

use eframe::egui;

use flir_env::Environment;
use flir_env::planck::RadianceLut;
use flir_scene::cpu::depth_to_scene;
use flir_sensor::Sensor;
use flir_sensor::cpu::{ScalarFrame, SensorState};
use flir_types::{Band, EnvironmentWire, SensorParams};
use infinity_viewports_host::camera::CameraLink;
use infinity_viewports_host::simvars::AircraftFeed;
use infinity_viewports_host::video::VideoReader;
use infinity_viewports_host::{Host, HostConfig, Sim, SimTarget};
use infinity_viewports_protocol::{
    APPLIED_FOLLOW, APPLIED_MISSING, APPLIED_PARENTED, APPLIED_PLACED, CameraControl, CameraStatus,
    MOUNT_AIRCRAFT, MOUNT_FOLLOW,
};

/// Role tags written by the bridge's stream writer (see `infinity_viewports_gpu::stream`).
const ROLE_DEPTH: u32 = 0;
const ROLE_COLOR: u32 = 2;

/// Base epoch (2024-06-21T00:00Z) the hour-of-day slider offsets from.
const BASE_UTC: f64 = 1_718_928_000.0;

/// A selectable streamed colour source: (label, DXGI format, occurrence index).
const COLOR_SOURCES: &[(&str, u32, u32)] = &[
    ("f26 R11G11B10 occ0", 26, 0),
    ("f26 R11G11B10 occ1", 26, 1),
    ("f10 R16G16B16A16 occ0", 10, 0),
    ("f10 R16G16B16A16 occ1", 10, 1),
    ("f26 occ2", 26, 2),
];

fn main() -> eframe::Result<()> {
    let mut bridge: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--bridge" {
            bridge = args.next().map(PathBuf::from);
        }
    }
    let Some(bridge_path) = bridge else {
        eprintln!("usage: flir-live --bridge <bridge.dll>");
        std::process::exit(2);
    };

    let host = match Host::new(HostConfig {
        bridge_path,
        bridge_sha256: None,
        sim_path: None,
    }) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("host init: {e}");
            std::process::exit(1);
        }
    };
    let sim = match host.attach(SimTarget::Auto) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("attach: {e}");
            std::process::exit(1);
        }
    };
    println!("attached to pid {}", sim.pid());

    // Enable the bridge, set an initial colour source, start the stream. The
    // develop is local; the bridge only streams.
    let _ = sim.flir().enable(true);
    let (_, fmt0, occ0) = COLOR_SOURCES[0];
    let _ = sim.set_stream_source(fmt0, occ0);
    match sim.start_stream() {
        Ok(()) => println!("stream start requested"),
        Err(e) => eprintln!("stream start: {e}"),
    }

    let app = LiveApp::new(host, sim);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1500.0, 760.0])
            .with_title("Infinity FLIR — live tuner"),
        ..Default::default()
    };
    eframe::run_native(
        "flir-live",
        options,
        Box::new(|_cc| Ok(Box::new(app) as Box<dyn eframe::App>)),
    )
}

struct LiveApp {
    _host: Host,
    sim: Sim,
    reader: Option<VideoReader>,
    params: SensorParams,
    env: EnvironmentWire,
    hour: f32,
    sensor_state: SensorState,
    lut_lwir: RadianceLut,
    lut_mwir: RadianceLut,
    // Latest streamed depth, kept so the thermal re-develops on every slider change.
    depth: Option<(usize, usize, Vec<f32>)>,
    color_tex: Option<egui::TextureHandle>,
    thermal_tex: Option<egui::TextureHandle>,
    color_src: usize,
    color_exposure: f32,
    last_seq: u64,
    frames: u64,
    status: String,
    cam: Option<CameraLink>,
    aircraft: AircraftFeed,
    cam_ready: bool,
    mount_aircraft: bool,
    right_m: f32,
    up_m: f32,
    forward_m: f32,
    yaw_deg: f32,
    pitch_deg: f32,
    roll_deg: f32,
    fov_deg: f32,
    near_m: f32,
    far_m: f32,
    aim_m: f32,
    match_far: bool,
    created_near: bool,
    auto_aim: bool,
    // Cloud compositing from the colour buffer. Volumetric clouds are composited
    // into the sim's nose *colour* RT but never write depth, so the depth-only
    // thermal sees straight through them. We detect cloud pixels in colour
    // (bright + desaturated = white) and composite cold cloud radiance into the
    // scene. `color_lin` holds the latest frame's linear RGB for this.
    color_lin: Option<(usize, usize, Vec<f32>)>,
    cloud_on: bool,
    cloud_thresh: f32,
    cloud_sat: f32,
    cloud_gain: f32,
    cloud_alt_m: f32,
}

impl LiveApp {
    fn new(host: Host, sim: Sim) -> LiveApp {
        LiveApp {
            _host: host,
            sim,
            reader: None,
            params: preset_params(),
            env: default_env(),
            hour: 13.0,
            sensor_state: SensorState::default(),
            lut_lwir: RadianceLut::build(Band::Lwir),
            lut_mwir: RadianceLut::build(Band::Mwir),
            depth: None,
            color_tex: None,
            thermal_tex: None,
            color_src: 0,
            color_exposure: 1.0,
            last_seq: 0,
            frames: 0,
            status: "connecting…".into(),
            cam: None,
            aircraft: AircraftFeed::new(),
            cam_ready: false,
            mount_aircraft: false,
            right_m: 0.0,
            up_m: 0.0,
            forward_m: 6.0,
            yaw_deg: 0.0,
            pitch_deg: 0.0,
            roll_deg: 0.0,
            fov_deg: 24.0,
            near_m: 0.05,
            far_m: 20_000.0,
            aim_m: 10_000.0,
            match_far: true,
            created_near: true,
            auto_aim: true,
            color_lin: None,
            cloud_on: true,
            cloud_thresh: 0.6,
            cloud_sat: 0.15,
            cloud_gain: 1.0,
            cloud_alt_m: 3000.0,
        }
    }

    /// Open the camera block once and, the first time it is readable, take the
    /// control the nose is already using so a fresh default does not clobber it.
    fn ensure_camera(&mut self) {
        if self.cam.is_none() {
            self.cam = self.sim.camera().ok();
        }
        if self.cam_ready {
            return;
        }
        let control = {
            let Some(cam) = &self.cam else { return };
            let Ok(control) = cam.read_control() else {
                return;
            };
            control
        };
        if control.valid() {
            self.apply_control(control);
        }
        self.cam_ready = true;
    }

    fn apply_control(&mut self, control: CameraControl) {
        let c = control.sanitized();
        self.mount_aircraft = c.mount == MOUNT_AIRCRAFT;
        self.right_m = c.right_m;
        self.up_m = c.up_m;
        self.forward_m = c.forward_m;
        self.yaw_deg = c.yaw_deg;
        self.pitch_deg = c.pitch_deg;
        self.roll_deg = c.roll_deg;
        self.fov_deg = c.fov_deg;
        self.created_near = c.near_m == 0.0;
        if c.near_m > 0.0 {
            self.near_m = c.near_m;
        }
        self.match_far = c.far_m == 0.0;
        if c.far_m > 0.0 {
            self.far_m = c.far_m;
        }
        self.auto_aim = c.aim_m == 0.0;
        if c.aim_m > 0.0 {
            self.aim_m = c.aim_m;
        }
    }

    fn camera_control(&self) -> CameraControl {
        CameraControl {
            mount: if self.mount_aircraft {
                MOUNT_AIRCRAFT
            } else {
                MOUNT_FOLLOW
            },
            right_m: self.right_m,
            up_m: self.up_m,
            forward_m: self.forward_m,
            yaw_deg: self.yaw_deg,
            pitch_deg: self.pitch_deg,
            roll_deg: self.roll_deg,
            fov_deg: self.fov_deg,
            near_m: if self.created_near { 0.0 } else { self.near_m },
            far_m: if self.match_far { 0.0 } else { self.far_m },
            aim_m: if self.auto_aim { 0.0 } else { self.aim_m },
            ..CameraControl::stock()
        }
        .sanitized()
    }

    fn push_camera(&mut self) {
        if !self.cam_ready {
            return;
        }
        let control = self.camera_control();
        let pose = self.aircraft.pose();
        if let Some(cam) = &self.cam {
            let _ = cam.write_control(control);
            let _ = cam.write_pose(pose);
        }
    }

    fn camera_controls(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.heading("Camera");
        if self.cam.is_none() {
            ui.label("waiting for the nose camera");
            return;
        }
        if let Some(cam) = &self.cam {
            match cam.read_status() {
                Ok(status) => {
                    ui.label(format_camera_status(&status));
                }
                Err(_) => {
                    ui.label("camera status unreadable");
                }
            }
        }
        let mut aircraft = self.mount_aircraft;
        egui::ComboBox::from_label("mount")
            .selected_text(if aircraft {
                "aircraft body"
            } else {
                "pilot view"
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut aircraft, false, "pilot view");
                ui.selectable_value(&mut aircraft, true, "aircraft body");
            });
        self.mount_aircraft = aircraft;
        ui.label("Aircraft holds the camera on the airframe. Looking around does not move it.");
        ui.label(self.aircraft.line());
        ui.add(egui::Slider::new(&mut self.right_m, -50.0..=50.0).text("right m"));
        ui.add(egui::Slider::new(&mut self.up_m, -20.0..=20.0).text("up m"));
        ui.add(egui::Slider::new(&mut self.forward_m, -10.0..=80.0).text("forward m"));
        ui.add(egui::Slider::new(&mut self.yaw_deg, -180.0..=180.0).text("yaw °"));
        ui.add(egui::Slider::new(&mut self.pitch_deg, -89.0..=89.0).text("pitch °"));
        ui.add(egui::Slider::new(&mut self.roll_deg, -180.0..=180.0).text("roll °"));
        ui.add(egui::Slider::new(&mut self.fov_deg, 5.0..=90.0).text("fov °"));
        ui.checkbox(&mut self.auto_aim, "automatic aim distance");
        ui.add_enabled(
            !self.auto_aim,
            egui::Slider::new(&mut self.aim_m, 10.0..=100_000.0)
                .text("aim m")
                .logarithmic(true),
        );
        ui.checkbox(&mut self.match_far, "match the pilot far plane");
        ui.add_enabled(
            !self.match_far,
            egui::Slider::new(&mut self.far_m, 10.0..=5_000_000.0)
                .text("far m")
                .logarithmic(true),
        );
        ui.checkbox(&mut self.created_near, "keep the created near plane");
        ui.add_enabled(
            !self.created_near,
            egui::Slider::new(&mut self.near_m, 0.01..=20.0).text("near m"),
        );
        if ui.button("reset camera").clicked() {
            self.apply_control(CameraControl::stock());
            self.near_m = 0.05;
            self.far_m = 20_000.0;
            self.aim_m = 10_000.0;
        }
    }

    fn lut(&self) -> &RadianceLut {
        if self.params.band_u8 == Band::Mwir as u8 {
            &self.lut_mwir
        } else {
            &self.lut_lwir
        }
    }

    /// Pull the newest streamed frame (if any): store depth, decode colour.
    fn poll_stream(&mut self, ctx: &egui::Context) {
        if self.reader.is_none() {
            self.reader = self.sim.open_video().ok();
        }
        let Some(reader) = &mut self.reader else {
            self.status = "waiting for video mapping…".into();
            return;
        };
        if let Some(frame) = reader.read_new() {
            self.last_seq = frame.seq;
            self.frames += 1;
            // Depth layer → stored for the thermal develop.
            if let Some(d) = frame.layers.iter().find(|l| l.role == ROLE_DEPTH) {
                let n = d.width as usize * d.height as usize;
                if d.bytes.len() >= n * 4 {
                    let depth: Vec<f32> = d.bytes[..n * 4]
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    self.depth = Some((d.width as usize, d.height as usize, depth));
                }
            }
            // Colour layer → reference view texture + linear RGB for cloud detect.
            if let Some(c) = frame.layers.iter().find(|l| l.role == ROLE_COLOR) {
                let (cw, ch) = (c.width as usize, c.height as usize);
                if let Some(img) = decode_color(c.format, cw, ch, &c.bytes, self.color_exposure) {
                    self.color_tex =
                        Some(ctx.load_texture("color", img, egui::TextureOptions::LINEAR));
                }
                if let Some(rgb) = decode_linear(c.format, cw, ch, &c.bytes) {
                    self.color_lin = Some((cw, ch, rgb));
                }
            }
            self.status = format!(
                "seq {} · {} frames · {} layers",
                frame.seq,
                self.frames,
                frame.layers.len()
            );
        }
    }

    /// Composite clouds into the scene radiance from the colour buffer. The sim
    /// renders volumetric clouds into the nose colour RT but they never write
    /// depth, so the depth-only scene develops straight through them. We flag
    /// cloud pixels as bright (relative to a high luminance percentile) AND
    /// desaturated (white — excludes blue sky and coloured terrain) and blend in
    /// cold cloud radiance at the cloud-altitude air temperature, obscuring what
    /// is behind — which is how a real LWIR sensor sees thick water cloud too.
    fn composite_clouds(&self, scene: &mut ScalarFrame, env: &Environment) {
        if !self.cloud_on {
            return;
        }
        let Some((cw, ch, rgb)) = &self.color_lin else {
            return;
        };
        if *cw != scene.w || *ch != scene.h {
            return; // colour/depth dimensions differ; skip rather than misalign
        }
        let mut samp: Vec<f32> = rgb
            .chunks_exact(3)
            .step_by(17)
            .map(|p| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2])
            .filter(|v| v.is_finite())
            .collect();
        let refl = percentile(&mut samp, 0.98).max(1e-4);
        let t_cloud = env.air_temp_at(self.cloud_alt_m);
        let l_cloud = self.lut().lookup(t_cloud);
        let (th, sg, gain) = (self.cloud_thresh, self.cloud_sat, self.cloud_gain);
        for i in 0..scene.w * scene.h {
            let (r, g, b) = (rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
            if !(r.is_finite() && g.is_finite() && b.is_finite()) {
                continue;
            }
            let maxc = r.max(g).max(b);
            let minc = r.min(g).min(b);
            let sat = if maxc > 1e-4 {
                (maxc - minc) / maxc
            } else {
                0.0
            };
            let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            let bright = smoothstep(th, th + 0.35, lum / refl);
            let white = 1.0 - smoothstep(sg, sg + 0.2, sat);
            let op = (bright * white * gain).clamp(0.0, 1.0);
            if op > 0.0 {
                scene.data[i] = op * l_cloud + (1.0 - op) * scene.data[i];
            }
        }
    }

    /// Re-develop the thermal image from the stored depth with the current knobs.
    fn develop_thermal(&mut self, ctx: &egui::Context) {
        let Some((w, h, depth)) = &self.depth else {
            return;
        };
        self.env.utc_unix_s = BASE_UTC + self.hour as f64 * 3600.0;
        let env = Environment::from_wire(&self.env);
        let mut scene: ScalarFrame = depth_to_scene(depth, *w, *h, &env, self.lut());
        self.composite_clouds(&mut scene, &env);
        let img = self.sensor_state.process(&scene, &Sensor::new(self.params));
        let color = egui::ColorImage::from_rgba_unmultiplied([img.w, img.h], &img.rgba);
        self.thermal_tex = Some(ctx.load_texture("thermal", color, egui::TextureOptions::LINEAR));
    }
}

impl eframe::App for LiveApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.aircraft.poll();
        self.ensure_camera();
        self.poll_stream(ctx);
        self.develop_thermal(ctx);

        egui::SidePanel::left("controls")
            .resizable(true)
            .default_width(320.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Stream");
                    ui.label(&self.status);
                    let mut src = self.color_src;
                    egui::ComboBox::from_label("colour source")
                        .selected_text(COLOR_SOURCES[src].0)
                        .show_ui(ui, |ui| {
                            for (i, (label, _, _)) in COLOR_SOURCES.iter().enumerate() {
                                ui.selectable_value(&mut src, i, *label);
                            }
                        });
                    if src != self.color_src {
                        self.color_src = src;
                        let (_, fmt, occ) = COLOR_SOURCES[src];
                        let _ = self.sim.set_stream_source(fmt, occ);
                    }
                    ui.add(
                        egui::Slider::new(&mut self.color_exposure, 0.1..=8.0)
                            .text("colour exposure")
                            .logarithmic(true),
                    );

                    self.camera_controls(ui);

                    ui.separator();
                    ui.heading("Sensor");
                    let p = &mut self.params;
                    let mut agc = p.agc != 0;
                    ui.checkbox(&mut agc, "AGC (auto gain)");
                    p.agc = agc as u8;
                    if agc {
                        ui.add(egui::Slider::new(&mut p.agc_clip, 0.0..=0.2).text("AGC clip"));
                        ui.add(
                            egui::Slider::new(&mut p.agc_plateau, 0.0..=1.0).text("AGC plateau"),
                        );
                        ui.add(
                            egui::Slider::new(&mut p.agc_speed, 0.02..=1.0)
                                .text("AGC speed")
                                .logarithmic(true),
                        );
                    } else {
                        ui.add(
                            egui::Slider::new(&mut p.gain, 0.05..=8.0)
                                .text("gain")
                                .logarithmic(true),
                        );
                        ui.add(egui::Slider::new(&mut p.level, -2.0..=2.0).text("level"));
                    }
                    ui.add(
                        egui::Slider::new(&mut p.gamma, 0.3..=8.0)
                            .text("gamma")
                            .logarithmic(true),
                    );
                    ui.add(egui::Slider::new(&mut p.brightness, 0.05..=1.5).text("brightness"));

                    let mut detail = p.detail != 0;
                    ui.checkbox(&mut detail, "detail (DDE)");
                    p.detail = detail as u8;
                    if detail {
                        ui.add(
                            egui::Slider::new(&mut p.detail_gain, 0.0..=4.0).text("detail gain"),
                        );
                    }
                    let mut noise = p.noise != 0;
                    ui.checkbox(&mut noise, "noise");
                    p.noise = noise as u8;
                    if noise {
                        ui.add(egui::Slider::new(&mut p.noise_sigma, 0.0..=0.03).text("noise σ"));
                    }
                    let mut fpn = p.column_fpn != 0;
                    ui.checkbox(&mut fpn, "column FPN");
                    p.column_fpn = fpn as u8;

                    egui::ComboBox::from_label("palette")
                        .selected_text(
                            ["phosphor", "gray", "ironbow"]
                                .get(p.palette_u8 as usize)
                                .copied()
                                .unwrap_or("phosphor"),
                        )
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut p.palette_u8, 0, "phosphor");
                            ui.selectable_value(&mut p.palette_u8, 1, "gray");
                            ui.selectable_value(&mut p.palette_u8, 2, "ironbow");
                        });
                    let mut black_hot = p.polarity_u8 == 1;
                    ui.checkbox(&mut black_hot, "black-hot");
                    p.polarity_u8 = black_hot as u8;
                    egui::ComboBox::from_label("band")
                        .selected_text(if p.band_u8 == Band::Mwir as u8 {
                            "MWIR"
                        } else {
                            "LWIR"
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut p.band_u8, Band::Lwir as u8, "LWIR");
                            ui.selectable_value(&mut p.band_u8, Band::Mwir as u8, "MWIR");
                        });

                    ui.separator();
                    ui.heading("Clouds (from colour)");
                    ui.checkbox(&mut self.cloud_on, "composite clouds");
                    if self.cloud_on {
                        ui.add(
                            egui::Slider::new(&mut self.cloud_thresh, 0.1..=1.5)
                                .text("brightness thresh"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.cloud_sat, 0.02..=0.6).text("whiteness"),
                        );
                        ui.add(egui::Slider::new(&mut self.cloud_gain, 0.0..=2.0).text("opacity"));
                        ui.add(
                            egui::Slider::new(&mut self.cloud_alt_m, 0.0..=12_000.0)
                                .text("cloud alt m"),
                        );
                    }

                    ui.separator();
                    ui.heading("Environment");
                    let e = &mut self.env;
                    ui.add(
                        egui::Slider::new(&mut e.ambient_temp_c, -40.0..=50.0).text("ambient °C"),
                    );
                    ui.add(egui::Slider::new(&mut e.humidity_pct, 1.0..=100.0).text("humidity %"));
                    ui.add(
                        egui::Slider::new(&mut e.visibility_m, 200.0..=40_000.0)
                            .text("visibility m")
                            .logarithmic(true),
                    );
                    ui.add(
                        egui::Slider::new(&mut e.ground_alt_m, -100.0..=4000.0)
                            .text("ground alt m"),
                    );
                    ui.add(egui::Slider::new(&mut e.alt_m, 0.0..=12_000.0).text("aircraft alt m"));
                    ui.add(egui::Slider::new(&mut self.hour, 0.0..=24.0).text("hour (UTC)"));

                    ui.separator();
                    if ui.button("reset sensor + env").clicked() {
                        self.params = preset_params();
                        self.env = default_env();
                        self.hour = 13.0;
                        self.sensor_state.reset();
                    }
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            let avail = ui.available_size();
            let cell = egui::vec2((avail.x - 16.0) * 0.5, avail.y - 24.0);
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.label("colour (sim nose view)");
                    if let Some(t) = &self.color_tex {
                        ui.add(
                            egui::Image::new(t)
                                .fit_to_exact_size(cell)
                                .maintain_aspect_ratio(true),
                        );
                    } else {
                        ui.label("— no colour frame yet —");
                    }
                });
                ui.separator();
                ui.vertical(|ui| {
                    ui.label("thermal (developed here)");
                    if let Some(t) = &self.thermal_tex {
                        ui.add(
                            egui::Image::new(t)
                                .fit_to_exact_size(cell)
                                .maintain_aspect_ratio(true),
                        );
                    } else {
                        ui.label("— no depth frame yet —");
                    }
                });
            });
        });

        self.push_camera();
        // Keep developing continuously; the stream advances independently.
        ctx.request_repaint_after(std::time::Duration::from_millis(16));
    }
}

fn format_camera_status(status: &CameraStatus) -> String {
    let n = status
        .note
        .iter()
        .position(|&c| c == 0)
        .unwrap_or(status.note.len());
    let note = String::from_utf8_lossy(&status.note[..n]);
    if note.is_empty() && status.fov_deg == 0.0 && status.far_m == 0.0 {
        return "waiting for the nose driver".into();
    }
    let label = if note.is_empty() {
        applied_label(status.applied)
    } else {
        note.as_ref()
    };
    format!(
        "{label} · fov {:.1}° · near {:.2} m · far {:.0} m",
        status.fov_deg, status.near_m, status.far_m
    )
}

fn applied_label(applied: u32) -> &'static str {
    match applied {
        APPLIED_PLACED => "on the aircraft axes",
        APPLIED_PARENTED => "attached to the aircraft",
        APPLIED_MISSING => "aircraft body not found",
        APPLIED_FOLLOW => "following the pilot view",
        _ => "camera status unknown",
    }
}

/// Decode a streamed HDR colour layer to linear RGB floats (`w*h*3`), reusing
/// the same f11/f10/half unpacking as the display path. For cloud detection.
fn decode_linear(fmt: u32, w: usize, h: usize, bytes: &[u8]) -> Option<Vec<f32>> {
    let n = w * h;
    let mut rgb = vec![0f32; n * 3];
    match fmt {
        26 => {
            if bytes.len() < n * 4 {
                return None;
            }
            for i in 0..n {
                let u = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
                rgb[i * 3] = f11(u & 0x7ff);
                rgb[i * 3 + 1] = f11((u >> 11) & 0x7ff);
                rgb[i * 3 + 2] = f10(((u >> 22) & 0x3ff) as u16);
            }
        }
        10 => {
            if bytes.len() < n * 8 {
                return None;
            }
            for i in 0..n {
                let base = i * 8;
                for c in 0..3 {
                    let half = u16::from_le_bytes(
                        bytes[base + c * 2..base + c * 2 + 2].try_into().unwrap(),
                    );
                    rgb[i * 3 + c] = half_to_f32(half);
                }
            }
        }
        _ => return None,
    }
    Some(rgb)
}

/// The tuned default sensor look (clip 0.03, plateau 0.33, instant AGC, gamma
/// 5.5, brightness 0.52, strong DDE). Keeps band/palette/polarity/gain/level
/// from the protocol default (LWIR, phosphor, white-hot). Used for the app's
/// initial params and the "reset sensor + env" button.
fn preset_params() -> SensorParams {
    SensorParams {
        agc: 1,
        agc_clip: 0.030,
        agc_plateau: 0.33,
        agc_speed: 1.0,
        gamma: 5.5,
        brightness: 0.52,
        detail: 1,
        detail_gain: 2.75,
        noise: 1,
        noise_sigma: 0.0,
        column_fpn: 1,
        ..SensorParams::default()
    }
}

/// Smoothstep (Hermite) between `e0` and `e1`.
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0).abs().max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// `p`-quantile of `v` (sorts in place). `p` in [0,1].
fn percentile(v: &mut [f32], p: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((v.len() as f32 - 1.0) * p.clamp(0.0, 1.0)).round() as usize;
    v[idx]
}

/// Decode a streamed HDR colour layer to an 8-bit egui image (Reinhard + gamma,
/// auto-exposed from a downsampled 99.5th percentile, scaled by `exposure`).
fn decode_color(
    fmt: u32,
    w: usize,
    h: usize,
    bytes: &[u8],
    exposure: f32,
) -> Option<egui::ColorImage> {
    let n = w * h;
    let mut rgb = vec![0f32; n * 3];
    match fmt {
        26 => {
            if bytes.len() < n * 4 {
                return None;
            }
            for i in 0..n {
                let u = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
                rgb[i * 3] = f11(u & 0x7ff);
                rgb[i * 3 + 1] = f11((u >> 11) & 0x7ff);
                rgb[i * 3 + 2] = f10(((u >> 22) & 0x3ff) as u16);
            }
        }
        10 => {
            if bytes.len() < n * 8 {
                return None;
            }
            for i in 0..n {
                let base = i * 8;
                for c in 0..3 {
                    let half = u16::from_le_bytes(
                        bytes[base + c * 2..base + c * 2 + 2].try_into().unwrap(),
                    );
                    rgb[i * 3 + c] = half_to_f32(half);
                }
            }
        }
        _ => return None,
    }

    // Auto-exposure: 99.5th percentile of a downsampled luminance.
    let mut sample: Vec<f32> = rgb
        .chunks_exact(3)
        .step_by(17)
        .map(|p| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2])
        .collect();
    sample.retain(|v| v.is_finite());
    let scale = if sample.is_empty() {
        1.0
    } else {
        sample.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let k = ((sample.len() as f32 * 0.995) as usize).min(sample.len() - 1);
        sample[k].max(1e-4)
    };

    let mut px = vec![0u8; n * 4];
    for i in 0..n {
        for c in 0..3 {
            let mut x = rgb[i * 3 + c] * exposure / scale;
            if !x.is_finite() {
                x = 0.0;
            }
            let x = x / (1.0 + x); // Reinhard
            px[i * 4 + c] = ((x.clamp(0.0, 1.0)).powf(1.0 / 2.2) * 255.0) as u8;
        }
        px[i * 4 + 3] = 255;
    }
    Some(egui::ColorImage::from_rgba_unmultiplied([w, h], &px))
}

/// 11-bit float (5-bit exp, 6-bit mantissa, no sign) → f32.
fn f11(u: u32) -> f32 {
    let e = (u >> 6) & 0x1f;
    let m = u & 0x3f;
    if e == 0 {
        m as f32 / 64.0 * 2f32.powi(-14)
    } else {
        (1.0 + m as f32 / 64.0) * 2f32.powi(e as i32 - 15)
    }
}

/// 10-bit float (5-bit exp, 5-bit mantissa, no sign) → f32.
fn f10(u: u16) -> f32 {
    let e = (u >> 5) & 0x1f;
    let m = u & 0x1f;
    if e == 0 {
        m as f32 / 32.0 * 2f32.powi(-14)
    } else {
        (1.0 + m as f32 / 32.0) * 2f32.powi(e as i32 - 15)
    }
}

/// IEEE half (16-bit, signed) → f32.
fn half_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let v = if exp == 0 {
        (mant as f32 / 1024.0) * 2f32.powi(-14)
    } else if exp == 0x1f {
        if mant == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + mant as f32 / 1024.0) * 2f32.powi(exp as i32 - 15)
    };
    if sign == 1 { -v } else { v }
}

/// A plausible clear daytime environment; the sliders tune from here.
fn default_env() -> EnvironmentWire {
    EnvironmentWire {
        ambient_temp_c: 15.0,
        pressure_hpa: 1013.0,
        humidity_pct: 55.0,
        dewpoint_c: f32::NAN,
        visibility_m: 20_000.0,
        precip_kind: 0,
        in_cloud: 0,
        _pad: [0; 2],
        precip_rate: 0.0,
        wind_dir_deg: 0.0,
        wind_speed_ms: 0.0,
        lat_deg: 47.6,
        lon_deg: -122.3,
        alt_m: 600.0,
        ground_alt_m: 100.0,
        utc_unix_s: BASE_UTC + 13.0 * 3600.0,
        sun_dir: [f32::NAN; 3],
    }
}
