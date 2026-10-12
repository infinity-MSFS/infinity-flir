use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::flir::{Crop, FlirSettings, SourceFrame};
use crate::warp::Warp;

type Notify = Box<dyn Fn() + Send + Sync>;

struct Shared {
    latest: Mutex<Option<SourceFrame>>,
    ready: Condvar,
    /// Output width, the render viewport within the capture, and the warp
    /// that replaces the viewport's rectangle when the host has one.
    geometry: Mutex<(usize, Crop, Option<Warp>)>,
    frames: AtomicU64,
    /// The last captured frame's size, pixels, before any crop or warp.
    size: Mutex<Option<(usize, usize)>>,
    notify: Mutex<Option<Notify>>,
}

impl Shared {
    fn publish(&self, frame: SourceFrame) {
        *self.latest.lock().unwrap() = Some(frame);
        self.frames.fetch_add(1, Ordering::Relaxed);
        self.ready.notify_all();
        if let Some(n) = &*self.notify.lock().unwrap() {
            n();
        }
    }
}

pub struct Feed {
    shared: Arc<Shared>,
    _handle: imp::CaptureHandle,
}

impl Feed {
    pub fn start(target: &Target, s: &FlirSettings) -> Result<Self, String> {
        let shared = Arc::new(Shared {
            latest: Mutex::new(None),
            ready: Condvar::new(),
            geometry: Mutex::new((s.sensor_width, s.crop, s.warp)),
            frames: AtomicU64::new(0),
            size: Mutex::new(None),
            notify: Mutex::new(None),
        });
        let handle = imp::start(target, shared.clone(), s.capture_hdr)?;
        Ok(Self {
            shared,
            _handle: handle,
        })
    }

    pub fn configure(&self, s: &FlirSettings) {
        *self.shared.geometry.lock().unwrap() = (s.sensor_width, s.crop, s.warp);
    }

    pub fn on_frame(&self, f: impl Fn() + Send + Sync + 'static) {
        *self.shared.notify.lock().unwrap() = Some(Box::new(f));
    }

    pub fn latest(&self) -> Option<SourceFrame> {
        self.shared.latest.lock().unwrap().take()
    }

    pub fn wait(&self, timeout: Duration) -> Option<SourceFrame> {
        let guard = self.shared.latest.lock().unwrap();
        let (mut guard, _) = self
            .shared
            .ready
            .wait_timeout_while(guard, timeout, |f| f.is_none())
            .unwrap();
        guard.take()
    }

    pub fn frames(&self) -> u64 {
        self.shared.frames.load(Ordering::Relaxed)
    }

    /// The captured window's size, pixels, once a frame has arrived — what
    /// a host needs to build a [`Warp`] for its viewport.
    pub fn source_size(&self) -> Option<(usize, usize)> {
        *self.shared.size.lock().unwrap()
    }
}

pub fn find_window(title: &str) -> Option<Target> {
    list_targets()
        .into_iter()
        .find(|(name, t)| matches!(t, Target::Window(_)) && name.contains(title))
        .map(|(_, t)| t)
}

#[cfg(windows)]
mod imp {
    use super::Shared;
    use crate::flir::{PixelFormat, downsample_raw, downsample_warped};
    use std::sync::Arc;
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
                    format!("Monitor {} {dev} ({w}x{h})", i + 1),
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
        fmt: PixelFormat,
    }

    impl GraphicsCaptureApiHandler for Handler {
        type Flags = (Arc<Shared>, PixelFormat);
        type Error = BoxErr;

        fn new(ctx: Context<Self::Flags>) -> Result<Self, Self::Error> {
            let (shared, fmt) = ctx.flags;
            Ok(Self { shared, fmt })
        }

        fn on_frame_arrived(
            &mut self,
            frame: &mut Frame,
            _control: InternalCaptureControl,
        ) -> Result<(), Self::Error> {
            let (out_w, crop, warp) = *self.shared.geometry.lock().unwrap();
            let mut buf = frame.buffer()?;
            let (w, h, pitch) = (
                buf.width() as usize,
                buf.height() as usize,
                buf.row_pitch() as usize,
            );
            *self.shared.size.lock().unwrap() = Some((w, h));
            let raw = buf.as_raw_buffer();
            let sf = match warp {
                Some(warp) => downsample_warped(raw, w, h, pitch, self.fmt, out_w, crop, &warp),
                None => downsample_raw(raw, w, h, pitch, self.fmt, out_w, crop),
            };
            self.shared.publish(sf);
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

    pub fn start(target: &Target, shared: Arc<Shared>, hdr: bool) -> Result<CaptureHandle, String> {
        let (fmt, color) = if hdr {
            (PixelFormat::Rgba16F, ColorFormat::Rgba16F)
        } else {
            (PixelFormat::Rgba8, ColorFormat::Rgba8)
        };
        let flags = (shared, fmt);
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
    use std::sync::Arc;

    #[derive(Clone)]
    pub enum Target {
        Window(std::convert::Infallible),
    }

    pub struct CaptureHandle;

    pub fn list_targets() -> Vec<(String, Target)> {
        Vec::new()
    }

    pub fn start(_: &Target, _: Arc<Shared>, _: bool) -> Result<CaptureHandle, String> {
        Err("screen capture is only available on Windows".into())
    }
}

pub use imp::{Target, list_targets};
