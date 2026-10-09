use std::sync::atomic::{AtomicU64, AtomicUsize};
use std::sync::{Arc, Mutex};

use crate::flir::SourceFrame;

pub struct Shared {
    pub latest: Mutex<Option<SourceFrame>>,
    pub sensor_w: AtomicUsize,
    pub frames: AtomicU64,
}

impl Shared {
    pub fn new(sensor_w: usize) -> Arc<Self> {
        Arc::new(Self {
            latest: Mutex::new(None),
            sensor_w: AtomicUsize::new(sensor_w),
            frames: AtomicU64::new(0),
        })
    }
}

#[cfg(windows)]
mod imp {
    use super::Shared;
    use crate::flir::{PixelFormat, downsample_raw};
    use eframe::egui;
    use std::sync::Arc;
    use std::sync::atomic::Ordering;
    use windows_capture::capture::{CaptureControl, Context, GraphicsCaptureApiHandler};
    use windows_capture::frame::Frame;
    use windows_capture::graphics_capture_api::InternalCaptureControl;
    use windows_capture::monitor::Monitor;
    use windows_capture::settings::{
        ColorFormat, CursorCaptureSettings, DirtyRegionSettings, DrawBorderSettings,
        MinimumUpdateIntervalSettings, SecondaryWindowSettings, Settings,
    };
    use windows_capture::window::Window;

    type BoxErr = Box<dyn std::error::Error + Send + Sync>;

    #[derive(Clone)]
    pub enum Target {
        Monitor(Monitor),
        Window(Window),
    }

    pub fn list_targets() -> Vec<(String, Target)> {
        let mut out = Vec::new();
        if let Ok(mons) = Monitor::enumerate() {
            for (i, m) in mons.into_iter().enumerate() {
                let dev = m.device_name().unwrap_or_default();
                let (w, h) = (m.width().unwrap_or(0), m.height().unwrap_or(0));
                out.push((
                    format!("Monitor {} {dev} ({w}x{h}", i + 1),
                    Target::Monitor(m),
                ));
            }
        }
        if let Ok(wins) = Window::enumerate() {
            let mut wins: Vec<_> = wins
                .into_iter()
                .filter_map(|w| {
                    let t = w.title().ok()?;
                    (!t.trim().is_empty()).then_some((t, w))
                })
                .collect();
            wins.sort_by_key(|(t, _)| !t.contains("Flight Simulator"));
            for (t, w) in wins {
                out.push((format!("Window: {t}"), Target::Window(w)));
            }
        }
        out
    }

    struct Handler {
        shared: Arc<Shared>,
        ctx: egui::Context,
        fmt: PixelFormat,
    }

    impl GraphicsCaptureApiHandler for Handler {
        type Flags = (Arc<Shared>, egui::Context, PixelFormat);
        type Error = BoxErr;

        fn new(ctx: Context<Self::Flags>) -> Result<Self, Self::Error> {
            let (shared, ctx, fmt) = ctx.flags;
            Ok(Self { shared, ctx, fmt })
        }

        fn on_frame_arrived(
            &mut self,
            frame: &mut Frame,
            _control: InternalCaptureControl,
        ) -> Result<(), Self::Error> {
            let out_w = self.shared.sensor_w.load(Ordering::Relaxed);
            let mut buf = frame.buffer()?;
            let (w, h, pitch) = (
                buf.width() as usize,
                buf.height() as usize,
                buf.row_pitch() as usize,
            );
            let sf = downsample_raw(buf.as_raw_buffer(), w, h, pitch, self.fmt, out_w);
            *self.shared.latest.lock().unwrap() = Some(sf);
            self.shared.frames.fetch_add(1, Ordering::Relaxed);
            self.ctx.request_repaint();
            Ok(())
        }

        fn on_closed(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    pub struct CaptureHandle(Option<CaptureControl<Handler, BoxErr>>);

    impl Drop for CaptureHandle {
        fn drop(&mut self) {
            if let Some(c) = self.0.take() {
                let _ = c.stop();
            }
        }
    }

    pub fn start(
        target: &Target,
        shared: Arc<Shared>,
        ctx: egui::Context,
        hdr: bool,
    ) -> Result<CaptureHandle, String> {
        let (fmt, color) = if hdr {
            (PixelFormat::Rgba16F, ColorFormat::Rgba16F)
        } else {
            (PixelFormat::Rgba8, ColorFormat::Rgba8)
        };
        let flags = (shared, ctx, fmt);
        macro_rules! go {
            ($item:expr) => {
                Handler::start_free_threaded(Settings::new(
                    $item,
                    CursorCaptureSettings::WithoutCursor,
                    DrawBorderSettings::WithoutBorder,
                    SecondaryWindowSettings::Default,
                    MinimumUpdateIntervalSettings::Default,
                    DirtyRegionSettings::Default,
                    color,
                    flags,
                ))
            };
        }
        let control = match target {
            Target::Monitor(m) => go!(*m),
            Target::Window(w) => go!(*w),
        }
        .map_err(|e| format!("capture failed to start: {e:?}"))?;
        Ok(CaptureHandle(Some(control)))
    }
}

#[cfg(not(windows))]
mod imp {
    use super::Shared;
    use eframe::egui;
    use std::sync::Arc;

    #[derive(Clone)]
    pub enum Target {}

    pub struct CaptureHandle;

    pub fn list_targets() -> Vec<(String, Target)> {
        Vec::new()
    }

    pub fn start(
        _: &Target,
        _: Arc<Shared>,
        _: egui::Context,
        _: bool,
    ) -> Result<CaptureHandle, String> {
        Err("screen capture is only available on Windows".into())
    }
}

pub use imp::{CaptureHandle, Target, list_targets, start};
