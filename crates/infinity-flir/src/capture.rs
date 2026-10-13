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
    /// Where the window's CLIENT area sits in the captured frame. A
    /// windowed sim's capture carries its title bar and borders, and every
    /// crop and warp is of the client area, which is what the game renders.
    /// The whole frame for a monitor or a borderless window.
    client: Mutex<Crop>,
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
            client: Mutex::new(Crop::default()),
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

    /// `crop` as the capture applies it: inside the window's client area.
    /// A host sizing a [`Warp`] for the viewport takes the viewport's pixels
    /// from this, not from `crop` alone.
    pub fn viewport(&self, crop: Crop) -> Crop {
        crop.within(*self.shared.client.lock().unwrap())
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
    use crate::flir::{Crop, PixelFormat, downsample_raw, downsample_warped};
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

    /// Frames between re-reads of the window's client area: a window is
    /// resized rarely, and three Win32 calls a frame are not free.
    const CLIENT_EVERY_FRAMES: u64 = 30;

    /// Where `hwnd`'s client area lies in a `frame_w × frame_h` capture of
    /// it, as a fraction of the frame. WGC captures the window's visible
    /// bounds (`DWMWA_EXTENDED_FRAME_BOUNDS`): title bar and borders
    /// included. `None` when the window cannot be asked or the answer does
    /// not fit the frame — the caller then keeps the whole frame.
    fn client_crop(hwnd: usize, frame_w: usize, frame_h: usize) -> Option<Crop> {
        use windows::Win32::Foundation::{HWND, POINT, RECT};
        use windows::Win32::Graphics::Dwm::{DWMWA_EXTENDED_FRAME_BOUNDS, DwmGetWindowAttribute};
        use windows::Win32::Graphics::Gdi::ClientToScreen;
        use windows::Win32::UI::WindowsAndMessaging::GetClientRect;

        let hwnd = HWND(hwnd as *mut std::ffi::c_void);
        let mut bounds = RECT::default();
        let mut client = RECT::default();
        let mut origin = POINT { x: 0, y: 0 };
        // SAFETY: plain Win32 queries on a window handle WGC is capturing;
        // each writes only the struct it is handed.
        unsafe {
            DwmGetWindowAttribute(
                hwnd,
                DWMWA_EXTENDED_FRAME_BOUNDS,
                (&mut bounds as *mut RECT).cast(),
                std::mem::size_of::<RECT>() as u32,
            )
            .ok()?;
            GetClientRect(hwnd, &mut client).ok()?;
            if !ClientToScreen(hwnd, &mut origin).as_bool() {
                return None;
            }
        }
        let (bw, bh) = (bounds.right - bounds.left, bounds.bottom - bounds.top);
        if bw <= 0 || bh <= 0 || frame_w == 0 || frame_h == 0 {
            return None;
        }
        // The capture may be scaled from the bounds (DPI); scale with it.
        let sx = frame_w as f64 / f64::from(bw);
        let sy = frame_h as f64 / f64::from(bh);
        let x = f64::from(origin.x - bounds.left) * sx;
        let y = f64::from(origin.y - bounds.top) * sy;
        let w = f64::from(client.right - client.left) * sx;
        let h = f64::from(client.bottom - client.top) * sy;
        let (fw, fh) = (frame_w as f64, frame_h as f64);
        const SLACK_PX: f64 = 2.0;
        let fits = x >= -SLACK_PX
            && y >= -SLACK_PX
            && x + w <= fw + SLACK_PX
            && y + h <= fh + SLACK_PX
            && w > fw * 0.5
            && h > fh * 0.5;
        fits.then(|| Crop {
            x: (x.max(0.0) / fw) as f32,
            y: (y.max(0.0) / fh) as f32,
            w: (w.min(fw) / fw) as f32,
            h: (h.min(fh) / fh) as f32,
        })
    }

    struct Handler {
        shared: Arc<Shared>,
        fmt: PixelFormat,
        /// The window being captured, for its client area; `None` for a
        /// monitor.
        hwnd: Option<usize>,
        seen: u64,
    }

    impl GraphicsCaptureApiHandler for Handler {
        type Flags = (Arc<Shared>, PixelFormat, Option<usize>);
        type Error = BoxErr;

        fn new(ctx: Context<Self::Flags>) -> Result<Self, Self::Error> {
            let (shared, fmt, hwnd) = ctx.flags;
            Ok(Self {
                shared,
                fmt,
                hwnd,
                seen: 0,
            })
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
            if let Some(hwnd) = self.hwnd
                && self.seen.is_multiple_of(CLIENT_EVERY_FRAMES)
            {
                *self.shared.client.lock().unwrap() = client_crop(hwnd, w, h).unwrap_or_default();
            }
            self.seen = self.seen.wrapping_add(1);
            let crop = crop.within(*self.shared.client.lock().unwrap());
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
        let hwnd = match target {
            Target::Window(w) => Some(w.as_raw_hwnd() as usize),
            Target::Monitor(_) => None,
        };
        let flags = (shared, fmt, hwnd);
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
