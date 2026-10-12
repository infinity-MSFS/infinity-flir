//! A projective resample: the output picture is not a rectangle of the
//! capture but whatever region of it a 3×3 homography names.
//!
//! The MD-10's HUD is collimated — its picture sits at infinity, fixed to
//! the aeroplane — so the part of the sim's window that lies behind it is a
//! quadrilateral that moves with the pilot's view. Mapping the HUD's own
//! texture square through the camera onto the window is exactly projective,
//! so one [`Warp`] per frame resamples the window onto the HUD's pixel grid
//! and the picture overlies the world it shows.
//!
//! The library knows nothing about cameras: the host builds the matrix
//! (`infinity-manager-v3`'s `flir` crate does, from the gauge's camera
//! report) or hands four corners to [`Warp::from_corners`].

use serde::{Deserialize, Serialize};

/// Below this the projective divisor is the plane at infinity or behind
/// the eye: the point is not on the screen.
const MIN_W: f64 = 1e-9;

/// Output `(u, v)` (unit square, origin top left, `v` down) to source
/// `(x, y)` in the same convention:
/// `[x·w, y·w, w] = h · [u, v, 1]`, the point valid only for `w > 0`.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
pub struct Warp {
    /// Row-major.
    pub h: [[f64; 3]; 3],
    /// The output's height over its width; the picture is `out_w` wide and
    /// `out_w · aspect` tall.
    pub aspect: f32,
}

impl Warp {
    pub const IDENTITY: Self = Self {
        h: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        aspect: 1.0,
    };

    /// The source point output `(u, v)` samples, `None` when it lies at or
    /// behind the eye or is not finite.
    pub fn map(&self, u: f32, v: f32) -> Option<(f32, f32)> {
        let (u, v) = (f64::from(u), f64::from(v));
        let r = |i: usize| self.h[i][0] * u + self.h[i][1] * v + self.h[i][2];
        let w = r(2);
        // `is_finite` first: an ordered compare on a NaN raises INVALID.
        if !w.is_finite() || w <= MIN_W {
            return None;
        }
        let (x, y) = (r(0) / w, r(1) / w);
        (x.is_finite() && y.is_finite()).then_some((x as f32, y as f32))
    }

    /// The homography taking the output's corners — top left, top right,
    /// bottom right, bottom left — to `corners`, the source points they
    /// should land on. `None` when three of them are collinear.
    pub fn from_corners(corners: [(f32, f32); 4], aspect: f32) -> Option<Self> {
        // Heckbert's square-to-quad: closed form, no solver.
        let [(x0, y0), (x1, y1), (x2, y2), (x3, y3)] =
            corners.map(|(x, y)| (f64::from(x), f64::from(y)));
        let (sx, sy) = (x0 - x1 + x2 - x3, y0 - y1 + y2 - y3);
        let (dx1, dy1) = (x1 - x2, y1 - y2);
        let (dx2, dy2) = (x3 - x2, y3 - y2);
        let den = dx1 * dy2 - dx2 * dy1;
        if !den.is_finite() || den.abs() < MIN_W {
            return None;
        }
        let g = (sx * dy2 - dx2 * sy) / den;
        let hh = (dx1 * sy - sx * dy1) / den;
        let h = [
            [x1 - x0 + g * x1, x3 - x0 + hh * x3, x0],
            [y1 - y0 + g * y1, y3 - y0 + hh * y3, y0],
            [g, hh, 1.0],
        ];
        h.iter()
            .flatten()
            .all(|v| v.is_finite())
            .then_some(Self { h, aspect })
    }

    /// The output's height for a width, at least one pixel.
    pub fn out_h(&self, out_w: usize) -> usize {
        let a = if self.aspect.is_finite() && self.aspect > 0.0 {
            self.aspect
        } else {
            1.0
        };
        ((out_w as f32 * a).round() as usize).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: (f32, f32), b: (f32, f32)) -> bool {
        (a.0 - b.0).abs() < 1e-5 && (a.1 - b.1).abs() < 1e-5
    }

    #[test]
    fn the_corners_land_where_they_were_asked_to() {
        let c = [(0.3, 0.2), (0.7, 0.25), (0.65, 0.8), (0.32, 0.75)];
        let w = Warp::from_corners(c, 1.0).unwrap();
        for (uv, want) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
            .into_iter()
            .zip(c)
        {
            let got = w.map(uv.0, uv.1).unwrap();
            assert!(close(got, want), "{uv:?}: {got:?} vs {want:?}");
        }
    }

    #[test]
    fn a_rectangle_is_an_affine_crop() {
        let w = Warp::from_corners([(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)], 2.0).unwrap();
        assert!(close(w.map(0.5, 0.5).unwrap(), (0.75, 0.5)));
        assert_eq!(w.out_h(100), 200);
    }

    #[test]
    fn behind_the_eye_and_garbage_are_none_not_a_trap() {
        let mut w = Warp::IDENTITY;
        w.h[2] = [0.0, 0.0, -1.0];
        assert_eq!(w.map(0.5, 0.5), None);
        w.h[2] = [0.0, 0.0, f64::NAN];
        assert_eq!(w.map(0.5, 0.5), None);
        assert!(Warp::from_corners([(0.0, 0.0); 4], 1.0).is_none());
    }
}
