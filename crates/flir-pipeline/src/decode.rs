//! The stream's raw layer bytes → floats (moved from flir-live).

/// DXGI formats the nose colour layer arrives in.
pub const DXGI_R11G11B10_FLOAT: u32 = 26;
pub const DXGI_R16G16B16A16_FLOAT: u32 = 10;
/// The depth layer's format, R32_FLOAT.
pub const DXGI_R32_FLOAT: u32 = 41;

/// The stream's layer roles.
pub const ROLE_DEPTH: u32 = 0;
pub const ROLE_COLOR: u32 = 2;

/// A little-endian R32F layer, `w × h`.
pub fn depth_f32(w: usize, h: usize, bytes: &[u8]) -> Option<Vec<f32>> {
    let n = w * h;
    if bytes.len() < n * 4 {
        return None;
    }
    Some(
        bytes[..n * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

/// An HDR colour layer → linear RGB, `w × h × 3`. `None` for a format it
/// does not know or a short buffer.
pub fn linear_rgb(fmt: u32, w: usize, h: usize, bytes: &[u8]) -> Option<Vec<f32>> {
    let n = w * h;
    let mut rgb = vec![0f32; n * 3];
    match fmt {
        DXGI_R11G11B10_FLOAT => {
            if bytes.len() < n * 4 {
                return None;
            }
            for i in 0..n {
                let b = &bytes[i * 4..i * 4 + 4];
                let u = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                rgb[i * 3] = f11(u & 0x7ff);
                rgb[i * 3 + 1] = f11((u >> 11) & 0x7ff);
                rgb[i * 3 + 2] = f10(((u >> 22) & 0x3ff) as u16);
            }
        }
        DXGI_R16G16B16A16_FLOAT => {
            if bytes.len() < n * 8 {
                return None;
            }
            for i in 0..n {
                let base = i * 8;
                for c in 0..3 {
                    let o = base + c * 2;
                    rgb[i * 3 + c] = half_to_f32(u16::from_le_bytes([bytes[o], bytes[o + 1]]));
                }
            }
        }
        _ => return None,
    }
    Some(rgb)
}

/// 11-bit float (5-bit exp, 6-bit mantissa, no sign) → f32.
pub fn f11(u: u32) -> f32 {
    let e = (u >> 6) & 0x1f;
    let m = u & 0x3f;
    if e == 0 {
        m as f32 / 64.0 * 2f32.powi(-14)
    } else {
        (1.0 + m as f32 / 64.0) * 2f32.powi(e as i32 - 15)
    }
}

/// 10-bit float (5-bit exp, 5-bit mantissa, no sign) → f32.
pub fn f10(u: u16) -> f32 {
    let e = (u >> 5) & 0x1f;
    let m = u & 0x1f;
    if e == 0 {
        m as f32 / 32.0 * 2f32.powi(-14)
    } else {
        (1.0 + m as f32 / 32.0) * 2f32.powi(e as i32 - 15)
    }
}

/// IEEE half (16-bit, signed) → f32.
pub fn half_to_f32(h: u16) -> f32 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_floats_decode() {
        // 1.0 in each packing: exponent 15, mantissa 0.
        assert_eq!(f11(15 << 6), 1.0);
        assert_eq!(f10(15 << 5), 1.0);
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xbc00), -1.0);
    }

    #[test]
    fn short_buffers_are_refused() {
        assert!(depth_f32(4, 4, &[0; 10]).is_none());
        assert!(linear_rgb(DXGI_R11G11B10_FLOAT, 4, 4, &[0; 10]).is_none());
        assert!(linear_rgb(99, 1, 1, &[0; 16]).is_none());
    }
}
