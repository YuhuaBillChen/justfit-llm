//! Exact side-input arithmetic for the already qualified M4 grouped graphs.
//! RoPE values are an offline, pinned reference table, avoiding a runtime NumPy
//! dependency or platform libm changes. No weights or graph recipe are changed.
use crate::{bad, Result};
use std::{fs, io::Write, path::Path};

const ROPE: &[u8] = include_bytes!("../data/rope-m4-reference-f32.bin");
pub fn grid(h: usize, w: usize) -> Result<()> {
    if h < 2
        || w < 2
        || h > 128
        || w > 128
        || !h.is_multiple_of(2)
        || !w.is_multiple_of(2)
        || h * w > 8192
    {
        return Err(bad("grid exceeds qualified ANE geometry"));
    }
    Ok(())
}
pub fn coordinates(h: usize, w: usize) -> Vec<(usize, usize)> {
    let mut points = Vec::with_capacity(h * w);
    for y in (0..h).step_by(2) {
        for x in (0..w).step_by(2) {
            for dy in 0..2 {
                for dx in 0..2 {
                    points.push((y + dy, x + dx))
                }
            }
        }
    }
    points
}
pub fn patchify(rgb: &[u8], h: usize, w: usize) -> Result<Vec<f32>> {
    grid(h, w)?;
    if rgb.len() != h * w * 768 {
        return Err(bad("invalid RGB payload length"));
    }
    let mut result = Vec::with_capacity(h * w * 1536);
    for (y, x) in coordinates(h, w) {
        for c in 0..3 {
            for _ in 0..2 {
                for py in 0..16 {
                    for px in 0..16 {
                        result.push(
                            rgb[((y * 16 + py) * (w * 16) + x * 16 + px) * 3 + c] as f32 / 127.5
                                - 1.,
                        );
                    }
                }
            }
        }
    }
    Ok(result)
}
pub fn npy_positions(path: &Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path).map_err(|_| bad("position table unavailable"))?;
    if bytes.len() < 16 || &bytes[..6] != b"\x93NUMPY" {
        return Err(bad("invalid position table"));
    }
    let (header, begin) = if bytes[6] == 1 {
        (
            u16::from_le_bytes(bytes[8..10].try_into().unwrap()) as usize,
            10,
        )
    } else if bytes[6] == 2 {
        (
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
            12,
        )
    } else {
        return Err(bad("unsupported npy version"));
    };
    let text = std::str::from_utf8(
        bytes
            .get(begin..begin + header)
            .ok_or_else(|| bad("short npy header"))?,
    )
    .map_err(|_| bad("invalid npy header"))?;
    if !text.contains("'<f4'")
        || !text.contains("'fortran_order': False")
        || !text.contains("(2304, 1152)")
    {
        return Err(bad("unexpected position table shape/type"));
    }
    let payload = &bytes[begin + header..];
    if payload.len() != 2304 * 1152 * 4 {
        return Err(bad("invalid position table size"));
    }
    let values = payload
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect::<Vec<_>>();
    if values.iter().any(|v| !v.is_finite()) {
        return Err(bad("non-finite position table"));
    }
    Ok(values)
}
pub fn positions(table: &[f32], h: usize, w: usize) -> Result<Vec<f32>> {
    grid(h, w)?;
    if table.len() != 2304 * 1152 {
        return Err(bad("unexpected position table size"));
    }
    let mut result = Vec::with_capacity(h * w * 1152);
    for (y, x) in coordinates(h, w) {
        let y = y as f64 * 47. / (h - 1) as f64;
        let x = x as f64 * 47. / (w - 1) as f64;
        let ly = y.floor() as usize;
        let lx = x.floor() as usize;
        let mut taps = [(0usize, 0f32); 4];
        for dy in 0..2 {
            for dx in 0..2 {
                taps[dy * 2 + dx] = (
                    ((ly + dy).min(47) * 48 + (lx + dx).min(47)) * 1152,
                    ((1. - (y - ly as f64 - dy as f64).abs()).max(0.)
                        * (1. - (x - lx as f64 - dx as f64).abs()).max(0.))
                        as f32,
                )
            }
        }
        for c in 0..1152 {
            let mut sum = 0f32;
            for (index, weight) in taps {
                sum += table[index + c] * weight
            }
            result.push(sum)
        }
    }
    Ok(result)
}
pub fn rope(h: usize, w: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    grid(h, w)?;
    let value = |sine: bool, pos: usize, freq: usize| {
        let offset = 16 + ((if sine { 128 * 18 } else { 0 }) + pos * 18 + freq) * 4;
        f32::from_le_bytes(ROPE[offset..offset + 4].try_into().unwrap())
    };
    let mut cos = Vec::with_capacity(h * w * 72);
    let mut sin = Vec::with_capacity(h * w * 72);
    for (y, x) in coordinates(h, w) {
        for _ in 0..2 {
            for p in [y, x] {
                for f in 0..18 {
                    cos.push(value(false, p, f));
                    sin.push(value(true, p, f));
                }
            }
        }
    }
    Ok((cos, sin))
}
pub fn bf16(values: &[f32]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        if !v.is_finite() {
            return Err(bad("non-finite ANE rows"));
        }
        let bits = v.to_bits();
        out.extend((((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16).to_le_bytes());
    }
    Ok(out)
}
pub fn probe(rgb: &Path, table: &Path, h: usize, w: usize, out: &Path) -> Result<()> {
    fs::create_dir(out).map_err(|_| bad("geometry output exists"))?;
    let raw = fs::read(rgb).map_err(|_| bad("RGB input unavailable"))?;
    let positions = positions(&npy_positions(table)?, h, w)?;
    let (cos, sin) = rope(h, w)?;
    for (name, values) in [
        ("patches", patchify(&raw, h, w)?),
        ("position", positions),
        ("cosine", cos),
        ("sine", sin),
    ] {
        let mut file = fs::File::create(out.join(format!("{name}.f32")))
            .map_err(|_| bad("geometry output unavailable"))?;
        for value in values {
            file.write_all(&value.to_le_bytes())
                .map_err(|_| bad("geometry write failed"))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn patch_order_and_repeated_time_planes() {
        let rgb = (0..4 * 768).map(|i| (i % 256) as u8).collect::<Vec<_>>();
        let p = patchify(&rgb, 2, 2).unwrap();
        assert_eq!(&p[..256], &p[256..512]);
        assert_eq!(p.len(), 4 * 1536);
    }
    #[test]
    fn bf16_ties_round_even_and_nonfinite_is_refused() {
        assert_eq!(
            bf16(&[f32::from_bits(0x3f808000), f32::from_bits(0x3f818000)]).unwrap(),
            [0x80, 0x3f, 0x82, 0x3f]
        );
        assert!(bf16(&[f32::NAN]).is_err());
    }
}
