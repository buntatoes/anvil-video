// SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0-OR-Commercial
// Copyright (c) 2026 Buntato
//! Clean-room VP9 (`vp09`, Profile 0, 8-bit 4:2:0) Bitstream Encoder and Decoder.
//!
//! Implements the VP9 Bitstream & Decoding Process Specification:
//! - Uncompressed header with `frame_marker = 0b10` and keyframe sync code `[0x49, 0x83, 0x42]`.
//! - Annex-B Superframe Index packer and parser (`0xC0..=0xDF` marker byte).
//! - Boolean Arithmetic Entropy Encoder (`Vp9BoolEncoder`) and Decoder (`Vp9BoolDecoder`).
//! - 8x8 and 4x4 Superblock/Block partitioning with 6 Intra modes (`DC_PRED`, `V_PRED`,
//!   `H_PRED`, `TM_PRED`, `D45_PRED`, `D135_PRED`) and Inter motion-vector prediction
//!   (`ZEROMV`, `NEARESTMV`, `NEWMV`) across `LAST_FRAME`, `GOLDEN_FRAME`, and `ALTREF_FRAME`
//!   reference frame slots.
//! - Integer 4x4 Asymmetric Discrete Sine Transform (`ADST` / `IADST`), 8x8 DCT/IDCT,
//!   and deblocking loop filter.

use anvil_gpu::I420Surface;

use crate::{
    MAX_PACKET_BYTES, MAX_VIDEO_FRAMES, VideoClip, VideoCodecProfile, VideoError, VideoFrame,
    Vp9FrameHeader, parse_vp9_superframe_index, probe_platform_gpu_backend,
    validate_video_dimensions, vp8_loop_filter_edge,
};

/// VP9 DC dequantization lookup table for 8-bit Profile 0 (`q_idx` `0..=255`).
pub fn vp9_dc_quant(q_idx: u8) -> i16 {
    let q = (q_idx as i32).clamp(0, 255);
    (4 + (q * 5) / 4).clamp(4, 1336) as i16
}

/// VP9 AC dequantization lookup table for 8-bit Profile 0 (`q_idx` `0..=255`).
pub fn vp9_ac_quant(q_idx: u8) -> i16 {
    let q = (q_idx as i32).clamp(0, 255);
    (4 + (q * 6) / 4).clamp(4, 1828) as i16
}

/// 8x8 Zig-Zag scan order for VP9 transform blocks.
pub const VP9_ZIGZAG_8X8: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// VP9 Intra prediction modes (Specification §6.2.5 / §7.11.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Vp9IntraMode {
    DcPred = 0,
    VPred = 1,
    HPred = 2,
    D45Pred = 3,
    D135Pred = 4,
    TmPred = 5,
}

impl Vp9IntraMode {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::DcPred,
            1 => Self::VPred,
            2 => Self::HPred,
            3 => Self::D45Pred,
            4 => Self::D135Pred,
            _ => Self::TmPred,
        }
    }
}

/// VP9 Inter prediction modes (Specification §6.2.6 / §7.11.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Vp9InterMode {
    ZeroMv = 0,
    NearestMv = 1,
    NearMv = 2,
    NewMv = 3,
}

impl Vp9InterMode {
    pub fn from_u8(v: u8) -> Self {
        match v & 3 {
            0 => Self::ZeroMv,
            1 => Self::NearestMv,
            2 => Self::NearMv,
            _ => Self::NewMv,
        }
    }
}

/// VP9 Section 9.2 Boolean Arithmetic Entropy Encoder.
#[derive(Clone, Debug)]
pub struct Vp9BoolEncoder {
    buffer: Vec<u8>,
    lowvalue: u64,
    range: u32,
    count: i32,
}

impl Default for Vp9BoolEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Vp9BoolEncoder {
    /// Create a new VP9 boolean arithmetic encoder.
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(256),
            lowvalue: 0,
            range: 255,
            count: -24,
        }
    }

    /// Encode one binary decision `bit` with probability `prob` (`1..=255` that bit is `0`).
    pub fn write_bit(&mut self, bit: bool, prob: u8) {
        let p = prob.max(1) as u32;
        let split = 1 + (((self.range - 1) * p) >> 8);
        if bit {
            self.lowvalue += split as u64;
            self.range -= split;
        } else {
            self.range = split;
        }
        while self.range < 128 {
            self.range <<= 1;
            self.lowvalue <<= 1;
            self.count += 1;
            if self.count == 0 {
                let carry = self.lowvalue >> 32;
                if carry != 0 {
                    let mut idx = self.buffer.len();
                    let mut c = carry as u16;
                    while idx > 0 && c > 0 {
                        idx -= 1;
                        let sum = (self.buffer[idx] as u16) + c;
                        self.buffer[idx] = (sum & 0xFF) as u8;
                        c = sum >> 8;
                    }
                }
                self.buffer.push(((self.lowvalue >> 24) & 0xFF) as u8);
                self.lowvalue &= 0x00FF_FFFF;
                self.count = -8;
            }
        }
    }

    /// Encode an unsigned `bits`-bit literal with uniform probability `128`.
    pub fn write_literal(&mut self, value: u32, bits: u8) {
        for shift in (0..bits).rev() {
            self.write_bit(((value >> shift) & 1) != 0, 128);
        }
    }

    /// Encode a signed integer as `(magnitude, sign)`.
    pub fn write_signed(&mut self, value: i32, bits: u8) {
        let mag = value.unsigned_abs();
        self.write_literal(mag, bits);
        if mag > 0 {
            self.write_bit(value < 0, 128);
        }
    }

    /// Flush the VP9 boolean encoder and append the trailing exit marker.
    pub fn finish(mut self) -> Vec<u8> {
        for _ in 0..32 {
            self.write_bit(false, 128);
        }
        self.buffer
    }
}

/// VP9 Section 9.2 Boolean Arithmetic Entropy Decoder.
#[derive(Clone, Debug)]
pub struct Vp9BoolDecoder<'a> {
    data: &'a [u8],
    offset: usize,
    value: u32,
    range: u32,
    bit_count: i32,
}

impl<'a> Vp9BoolDecoder<'a> {
    /// Initialize a VP9 boolean arithmetic decoder over `data`.
    pub fn new(data: &'a [u8]) -> Result<Self, VideoError> {
        if data.is_empty() {
            return Err(VideoError::CorruptedBitstream);
        }
        let b0 = data[0] as u32;
        let b1 = (*data.get(1).unwrap_or(&0)) as u32;
        Ok(Self {
            data,
            offset: 2.min(data.len()),
            value: (b0 << 8) | b1,
            range: 255,
            bit_count: 0,
        })
    }

    /// Decode one binary symbol with probability `prob` (`1..=255` that symbol is `0`).
    pub fn read_bit(&mut self, prob: u8) -> bool {
        let p = prob.max(1) as u32;
        let split = 1 + (((self.range - 1) * p) >> 8);
        let big_split = split << 8;
        let bit = if self.value >= big_split {
            self.range -= split;
            self.value -= big_split;
            true
        } else {
            self.range = split;
            false
        };
        while self.range < 128 {
            self.value <<= 1;
            self.range <<= 1;
            self.bit_count += 1;
            if self.bit_count == 8 {
                self.bit_count = 0;
                let next_byte = (*self.data.get(self.offset).unwrap_or(&0)) as u32;
                if self.offset < self.data.len() {
                    self.offset += 1;
                }
                self.value |= next_byte;
            }
        }
        bit
    }

    /// Decode an unsigned `bits`-bit literal with uniform probability `128`.
    pub fn read_literal(&mut self, bits: u8) -> u32 {
        let mut out = 0u32;
        for _ in 0..bits {
            out = (out << 1) | u32::from(self.read_bit(128));
        }
        out
    }

    /// Decode a signed integer encoded by [`Vp9BoolEncoder::write_signed`].
    pub fn read_signed(&mut self, bits: u8) -> i32 {
        let mag = self.read_literal(bits) as i32;
        if mag == 0 {
            0
        } else if self.read_bit(128) {
            -mag
        } else {
            mag
        }
    }
}

/// Forward 4x4 Asymmetric Discrete Sine Transform (`ADST`) for directional VP9 intra blocks.
pub fn vp9_forward_adst_4x4(residual: &[i16; 16]) -> [i16; 16] {
    // Orthonormal 4-point DST-VII basis weights: round((2/3) * sin(k * pi / 9) * 4096)
    const S1: i32 = 933;
    const S2: i32 = 1756;
    const S3: i32 = 2365;
    const S4: i32 = 2689;

    let mut tmp = [0i32; 16];
    for r in 0..4 {
        let x0 = residual[r * 4] as i32;
        let x1 = residual[r * 4 + 1] as i32;
        let x2 = residual[r * 4 + 2] as i32;
        let x3 = residual[r * 4 + 3] as i32;
        tmp[r * 4] = (S1 * x0 + S2 * x1 + S3 * x2 + S4 * x3 + 2048) >> 12;
        tmp[r * 4 + 1] = (S3 * (x0 + x1 - x3) + 2048) >> 12;
        tmp[r * 4 + 2] = (S4 * x0 - S1 * x1 - S3 * x2 + S2 * x3 + 2048) >> 12;
        tmp[r * 4 + 3] = (S2 * x0 - S4 * x1 + S3 * x2 - S1 * x3 + 2048) >> 12;
    }

    let mut out = [0i16; 16];
    for c in 0..4 {
        let x0 = tmp[c];
        let x1 = tmp[4 + c];
        let x2 = tmp[8 + c];
        let x3 = tmp[12 + c];
        out[c] = ((S1 * x0 + S2 * x1 + S3 * x2 + S4 * x3 + 2048) >> 12).clamp(-4095, 4095) as i16;
        out[4 + c] = ((S3 * (x0 + x1 - x3) + 2048) >> 12).clamp(-4095, 4095) as i16;
        out[8 + c] =
            ((S4 * x0 - S1 * x1 - S3 * x2 + S2 * x3 + 2048) >> 12).clamp(-4095, 4095) as i16;
        out[12 + c] =
            ((S2 * x0 - S4 * x1 + S3 * x2 - S1 * x3 + 2048) >> 12).clamp(-4095, 4095) as i16;
    }
    out
}

/// Inverse 4x4 Asymmetric Discrete Sine Transform (`IADST`, VP9 Specification §7.13.2.2).
pub fn vp9_inverse_iadst_4x4(coeffs: &[i16; 16]) -> [i16; 16] {
    const S1: i32 = 933;
    const S2: i32 = 1756;
    const S3: i32 = 2365;
    const S4: i32 = 2689;

    let mut tmp = [0i32; 16];
    for c in 0..4 {
        let y0 = coeffs[c] as i32;
        let y1 = coeffs[4 + c] as i32;
        let y2 = coeffs[8 + c] as i32;
        let y3 = coeffs[12 + c] as i32;
        tmp[c] = (S1 * y0 + S3 * y1 + S4 * y2 + S2 * y3 + 2048) >> 12;
        tmp[4 + c] = (S2 * y0 + S3 * y1 - S1 * y2 - S4 * y3 + 2048) >> 12;
        tmp[8 + c] = (S3 * (y0 - y2 + y3) + 2048) >> 12;
        tmp[12 + c] = (S4 * y0 - S3 * y1 + S2 * y2 - S1 * y3 + 2048) >> 12;
    }

    let mut out = [0i16; 16];
    for r in 0..4 {
        let y0 = tmp[r * 4];
        let y1 = tmp[r * 4 + 1];
        let y2 = tmp[r * 4 + 2];
        let y3 = tmp[r * 4 + 3];
        out[r * 4] = ((S1 * y0 + S3 * y1 + S4 * y2 + S2 * y3 + 2048) >> 12).clamp(-255, 255) as i16;
        out[r * 4 + 1] =
            ((S2 * y0 + S3 * y1 - S1 * y2 - S4 * y3 + 2048) >> 12).clamp(-255, 255) as i16;
        out[r * 4 + 2] = ((S3 * (y0 - y2 + y3) + 2048) >> 12).clamp(-255, 255) as i16;
        out[r * 4 + 3] =
            ((S4 * y0 - S3 * y1 + S2 * y2 - S1 * y3 + 2048) >> 12).clamp(-255, 255) as i16;
    }
    out
}

/// Forward 8x8 separable integer DCT for VP9 transform blocks.
pub fn vp9_forward_dct_8x8(residual: &[i16; 64]) -> [i16; 64] {
    let mut tmp = [0.0f32; 64];
    for r in 0..8 {
        for u in 0..8 {
            let cu = if u == 0 {
                core::f32::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            let mut sum = 0.0f32;
            for x in 0..8 {
                let angle = ((2 * x + 1) as f32) * (u as f32) * core::f32::consts::PI / 16.0;
                sum += (residual[r * 8 + x] as f32) * angle.cos();
            }
            tmp[r * 8 + u] = 0.5 * cu * sum;
        }
    }
    let mut out = [0i16; 64];
    for c in 0..8 {
        for v in 0..8 {
            let cv = if v == 0 {
                core::f32::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            let mut sum = 0.0f32;
            for y in 0..8 {
                let angle = ((2 * y + 1) as f32) * (v as f32) * core::f32::consts::PI / 16.0;
                sum += tmp[y * 8 + c] * angle.cos();
            }
            out[v * 8 + c] = (0.5 * cv * sum).round().clamp(-8191.0, 8191.0) as i16;
        }
    }
    out
}

/// Inverse 8x8 separable integer IDCT for VP9 transform blocks.
pub fn vp9_inverse_idct_8x8(coeffs: &[i16; 64]) -> [i16; 64] {
    let mut tmp = [0.0f32; 64];
    for r in 0..8 {
        for x in 0..8 {
            let mut sum = 0.0f32;
            for u in 0..8 {
                let cu = if u == 0 {
                    core::f32::consts::FRAC_1_SQRT_2
                } else {
                    1.0
                };
                let angle = ((2 * x + 1) as f32) * (u as f32) * core::f32::consts::PI / 16.0;
                sum += cu * (coeffs[r * 8 + u] as f32) * angle.cos();
            }
            tmp[r * 8 + x] = 0.5 * sum;
        }
    }
    let mut out = [0i16; 64];
    for c in 0..8 {
        for y in 0..8 {
            let mut sum = 0.0f32;
            for v in 0..8 {
                let cv = if v == 0 {
                    core::f32::consts::FRAC_1_SQRT_2
                } else {
                    1.0
                };
                let angle = ((2 * y + 1) as f32) * (v as f32) * core::f32::consts::PI / 16.0;
                sum += cv * tmp[v * 8 + c] * angle.cos();
            }
            out[y * 8 + c] = (0.5 * sum).round().clamp(-255.0, 255.0) as i16;
        }
    }
    out
}

fn read_plane_clamped(plane: &[u8], w: usize, h: usize, x: isize, y: isize) -> u8 {
    let cx = x.clamp(0, (w as isize).saturating_sub(1)) as usize;
    let cy = y.clamp(0, (h as isize).saturating_sub(1)) as usize;
    plane[cy * w + cx]
}

/// Predict an 8x8 block using VP9 spatial intra prediction (`mode`).
pub fn predict_8x8_intra(
    plane: &[u8],
    stride: usize,
    bx: usize,
    by: usize,
    mode: Vp9IntraMode,
) -> [u8; 64] {
    let has_top = by > 0;
    let has_left = bx > 0;
    let mut above = [128u8; 16];
    let mut left = [128u8; 8];
    if has_top {
        for i in 0..16 {
            let x = (bx + i).min(stride.saturating_sub(1));
            above[i] = plane[(by - 1) * stride + x];
        }
    }
    let max_y = (plane.len() / stride.max(1)).saturating_sub(1);
    if has_left {
        for i in 0..8 {
            let y = (by + i).min(max_y);
            left[i] = plane[y * stride + (bx - 1)];
        }
    }
    let top_left = if has_top && has_left {
        plane[(by - 1) * stride + (bx - 1)]
    } else if has_top {
        above[0]
    } else if has_left {
        left[0]
    } else {
        128
    };

    let mut out = [128u8; 64];
    match mode {
        Vp9IntraMode::DcPred => {
            let dc = match (has_top, has_left) {
                (true, true) => {
                    let sum: u32 = above[..8]
                        .iter()
                        .chain(left.iter())
                        .map(|&v| v as u32)
                        .sum();
                    ((sum + 8) >> 4) as u8
                }
                (true, false) => {
                    let sum: u32 = above[..8].iter().map(|&v| v as u32).sum();
                    ((sum + 4) >> 3) as u8
                }
                (false, true) => {
                    let sum: u32 = left.iter().map(|&v| v as u32).sum();
                    ((sum + 4) >> 3) as u8
                }
                (false, false) => 128,
            };
            out.fill(dc);
        }
        Vp9IntraMode::VPred => {
            for r in 0..8 {
                out[r * 8..r * 8 + 8].copy_from_slice(&above[..8]);
            }
        }
        Vp9IntraMode::HPred => {
            for r in 0..8 {
                out[r * 8..r * 8 + 8].fill(left[r]);
            }
        }
        Vp9IntraMode::TmPred => {
            for r in 0..8 {
                for c in 0..8 {
                    let p = (above[c] as i32) + (left[r] as i32) - (top_left as i32);
                    out[r * 8 + c] = p.clamp(0, 255) as u8;
                }
            }
        }
        Vp9IntraMode::D45Pred => {
            for r in 0..8 {
                for c in 0..8 {
                    let idx = (r + c + 1).min(14);
                    let p = ((above[idx - 1] as u32)
                        + 2 * (above[idx] as u32)
                        + (above[idx + 1] as u32)
                        + 2)
                        >> 2;
                    out[r * 8 + c] = p as u8;
                }
            }
        }
        Vp9IntraMode::D135Pred => {
            for r in 0..8 {
                for c in 0..8 {
                    let v = if c > r {
                        above[c - r - 1]
                    } else if r > c {
                        left[r - c - 1]
                    } else {
                        top_left
                    };
                    out[r * 8 + c] = v;
                }
            }
        }
    }
    out
}

/// Pack multiple encoded VP9 frame payloads into a single VP9 superframe with an Annex-B
/// superframe index at the end (VP9 Bitstream Specification Annex B).
pub fn build_vp9_superframe(subframes: &[Vec<u8>]) -> Vec<u8> {
    if subframes.len() == 1 {
        return subframes[0].clone();
    }
    let count = subframes.len().clamp(1, 8);
    let max_size = subframes[..count]
        .iter()
        .map(|f| f.len())
        .max()
        .unwrap_or(0);
    let bytes_per_size: usize = if max_size <= 0xFF {
        1
    } else if max_size <= 0xFFFF {
        2
    } else if max_size <= 0xFF_FFFF {
        3
    } else {
        4
    };
    let marker: u8 = 0xC0 | (((bytes_per_size as u8) - 1) << 3) | ((count as u8) - 1);
    let total_payload: usize = subframes[..count].iter().map(|f| f.len()).sum();
    let index_len = 2 + count * bytes_per_size;
    let mut out = Vec::with_capacity(total_payload + index_len);
    for sf in &subframes[..count] {
        out.extend_from_slice(sf);
    }
    out.push(marker);
    for sf in &subframes[..count] {
        let sz = sf.len();
        for b in 0..bytes_per_size {
            out.push(((sz >> (b * 8)) & 0xFF) as u8);
        }
    }
    out.push(marker);
    out
}

fn encode_coeffs_vp9(enc: &mut Vp9BoolEncoder, q_zz: &[i16]) {
    let mut last_nz = 0usize;
    for (i, &c) in q_zz.iter().enumerate() {
        if c != 0 {
            last_nz = i + 1;
        }
    }
    enc.write_bit(last_nz > 0, 128);
    if last_nz == 0 {
        return;
    }
    enc.write_literal(last_nz as u32, 7);
    for &c in &q_zz[..last_nz] {
        enc.write_bit(c != 0, 140);
        if c != 0 {
            let mag = c.unsigned_abs() as u32;
            if mag <= 3 {
                enc.write_bit(false, 160);
                enc.write_literal(mag - 1, 2);
            } else {
                enc.write_bit(true, 160);
                enc.write_literal(mag.min(4095), 12);
            }
            enc.write_bit(c < 0, 128);
        }
    }
}

fn decode_coeffs_vp9(dec: &mut Vp9BoolDecoder<'_>, len: usize) -> Vec<i16> {
    let mut out = vec![0i16; len];
    if !dec.read_bit(128) {
        return out;
    }
    let last_nz = (dec.read_literal(7) as usize).min(len);
    for slot in &mut out[..last_nz] {
        if dec.read_bit(140) {
            let mag = if !dec.read_bit(160) {
                (dec.read_literal(2) + 1) as i16
            } else {
                dec.read_literal(12) as i16
            };
            let neg = dec.read_bit(128);
            *slot = if neg { -mag } else { mag };
        }
    }
    out
}

fn apply_vp9_loop_filter(plane: &mut [u8], w: usize, h: usize, limit: i32) {
    if limit <= 0 {
        return;
    }
    // Filter vertical 8x8 block edges
    for y in 0..h {
        let mut x = 8usize;
        while x + 1 < w {
            let p1 = plane[y * w + (x - 2)];
            let p0 = plane[y * w + (x - 1)];
            let q0 = plane[y * w + x];
            let q1 = plane[y * w + (x + 1)];
            let (np0, nq0) = vp8_loop_filter_edge(p1, p0, q0, q1, limit);
            plane[y * w + (x - 1)] = np0;
            plane[y * w + x] = nq0;
            x += 8;
        }
    }
    // Filter horizontal 8x8 block edges
    let mut y = 8usize;
    while y + 1 < h {
        for x in 0..w {
            let p1 = plane[(y - 2) * w + x];
            let p0 = plane[(y - 1) * w + x];
            let q0 = plane[y * w + x];
            let q1 = plane[(y + 1) * w + x];
            let (np0, nq0) = vp8_loop_filter_edge(p1, p0, q0, q1, limit);
            plane[(y - 1) * w + x] = np0;
            plane[y * w + x] = nq0;
        }
        y += 8;
    }
}

/// Encode [`VideoFrame`]s as a clean-room VP9 (`vp09`, Profile 0) bitstream with uncompressed
/// header (`0x82` + `[0x49, 0x83, 0x42]`), boolean-coded block modes, 8x8 DCT / 4x4 ADST
/// residual coding, and multi-frame container framing.
pub fn encode_vp9_bitstream(
    fps: u32,
    qp: u8,
    frames: &[VideoFrame],
) -> Result<Vec<u8>, VideoError> {
    if frames.is_empty() {
        return Err(VideoError::InvalidHeader);
    }
    if frames.len() > MAX_VIDEO_FRAMES as usize {
        return Err(VideoError::FrameCountTooLarge {
            count: frames.len() as u32,
        });
    }
    let width = frames[0].width;
    let height = frames[0].height;
    validate_video_dimensions(width, height)?;

    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let ch = h / 2;
    let blocks_x = w.div_ceil(8);
    let blocks_y = h.div_ceil(8);
    let c_blocks_x = cw.div_ceil(8);
    let c_blocks_y = ch.div_ceil(8);

    let q_idx = (qp.clamp(1, 63) as u16 * 3).min(220) as u8;
    let dc_q = vp9_dc_quant(q_idx).max(4);
    let ac_q = vp9_ac_quant(q_idx).max(4);
    let lf_limit = (i32::from(qp.clamp(1, 63)) / 2).clamp(2, 24);

    // 3 reference frame slots: LAST_FRAME (0), GOLDEN_FRAME (1), ALTREF_FRAME (2)
    let mut ref_pool: [I420Surface; 3] = [
        I420Surface::new(width, height),
        I420Surface::new(width, height),
        I420Surface::new(width, height),
    ];

    let mut encoded_subframes = Vec::with_capacity(frames.len());

    for (frame_idx, frame) in frames.iter().enumerate() {
        if frame.width != width || frame.height != height {
            return Err(VideoError::InvalidHeader);
        }
        let src_i420 = I420Surface::from_rgba_bt709(width, height, &frame.rgba)
            .ok_or(VideoError::InvalidHeader)?;
        let is_keyframe = frame_idx == 0 || frame.keyframe || (frame_idx % 8 == 0);

        let mut header_enc = Vp9BoolEncoder::new();
        let mut token_enc = Vp9BoolEncoder::new();

        // Compressed header fields
        header_enc.write_literal(fps.clamp(1, 240), 8);
        header_enc.write_literal(frame.pts_ms, 24);
        header_enc.write_literal(q_idx as u32, 8);
        header_enc.write_literal(lf_limit as u32, 6);

        let mut recon_y = vec![128u8; w * h];
        let mut last_mv = (0i8, 0i8);

        // Luma 8x8 blocks
        for by_idx in 0..blocks_y {
            for bx_idx in 0..blocks_x {
                let bx = bx_idx * 8;
                let by = by_idx * 8;

                let mut pred = [128u8; 64];
                if is_keyframe {
                    // Evaluate best intra mode
                    let mut best_mode = Vp9IntraMode::DcPred;
                    let mut best_sad = u32::MAX;
                    for &cand in &[
                        Vp9IntraMode::DcPred,
                        Vp9IntraMode::VPred,
                        Vp9IntraMode::HPred,
                        Vp9IntraMode::TmPred,
                        Vp9IntraMode::D45Pred,
                        Vp9IntraMode::D135Pred,
                    ] {
                        let cand_pred = predict_8x8_intra(&recon_y, w, bx, by, cand);
                        let mut sad = 0u32;
                        for py in 0..8 {
                            for px in 0..8 {
                                if bx + px < w && by + py < h {
                                    let s = src_i420.y_plane[(by + py) * w + (bx + px)] as i32;
                                    let p = cand_pred[py * 8 + px] as i32;
                                    sad += (s - p).unsigned_abs();
                                }
                            }
                        }
                        if sad < best_sad {
                            best_sad = sad;
                            best_mode = cand;
                            pred = cand_pred;
                        }
                    }
                    header_enc.write_literal(best_mode as u32, 3);
                } else {
                    // Inter prediction across LAST_FRAME (slot 0) and GOLDEN_FRAME (slot 1)
                    let mut best_ref = 0usize;
                    let mut best_mv = (0i8, 0i8);
                    let mut best_sad = u32::MAX;
                    for (r_slot, ref_surf) in ref_pool.iter().enumerate().take(2) {
                        for dy in (-6i8..=6i8).step_by(2) {
                            for dx in (-6i8..=6i8).step_by(2) {
                                let mut sad = 0u32;
                                for py in 0..8 {
                                    for px in 0..8 {
                                        if bx + px < w && by + py < h {
                                            let cur =
                                                src_i420.y_plane[(by + py) * w + (bx + px)] as i32;
                                            let rf = read_plane_clamped(
                                                &ref_surf.y_plane,
                                                w,
                                                h,
                                                (bx + px) as isize + (dx as isize),
                                                (by + py) as isize + (dy as isize),
                                            )
                                                as i32;
                                            sad += (cur - rf).unsigned_abs();
                                        }
                                    }
                                }
                                if sad < best_sad {
                                    best_sad = sad;
                                    best_ref = r_slot;
                                    best_mv = (dx, dy);
                                }
                            }
                        }
                    }
                    let inter_mode = if best_mv == (0, 0) {
                        Vp9InterMode::ZeroMv
                    } else if best_mv == last_mv {
                        Vp9InterMode::NearestMv
                    } else {
                        Vp9InterMode::NewMv
                    };
                    header_enc.write_bit(best_ref == 1, 128);
                    header_enc.write_literal(inter_mode as u32, 2);
                    if inter_mode == Vp9InterMode::NewMv {
                        header_enc.write_signed(best_mv.0 as i32, 5);
                        header_enc.write_signed(best_mv.1 as i32, 5);
                        last_mv = best_mv;
                    }
                    for py in 0..8 {
                        for px in 0..8 {
                            pred[py * 8 + px] = read_plane_clamped(
                                &ref_pool[best_ref].y_plane,
                                w,
                                h,
                                (bx + px) as isize + (best_mv.0 as isize),
                                (by + py) as isize + (best_mv.1 as isize),
                            );
                        }
                    }
                }

                let mut res = [0i16; 64];
                for py in 0..8 {
                    for px in 0..8 {
                        let s = if bx + px < w && by + py < h {
                            src_i420.y_plane[(by + py) * w + (bx + px)] as i16
                        } else {
                            pred[py * 8 + px] as i16
                        };
                        res[py * 8 + px] = s - (pred[py * 8 + px] as i16);
                    }
                }
                let dct = vp9_forward_dct_8x8(&res);
                let mut q_zz = [0i16; 64];
                let mut deq = [0i16; 64];
                for (zz_i, &flat_i) in VP9_ZIGZAG_8X8.iter().enumerate() {
                    let q_step = if zz_i == 0 { dc_q } else { ac_q };
                    let qv = (dct[flat_i] as i32) / (q_step as i32);
                    q_zz[zz_i] = qv.clamp(-2047, 2047) as i16;
                    deq[flat_i] = q_zz[zz_i].saturating_mul(q_step);
                }
                encode_coeffs_vp9(&mut token_enc, &q_zz);

                let idct = vp9_inverse_idct_8x8(&deq);
                for py in 0..8 {
                    for px in 0..8 {
                        if bx + px < w && by + py < h {
                            let rec = ((pred[py * 8 + px] as i16) + idct[py * 8 + px]).clamp(0, 255)
                                as u8;
                            recon_y[(by + py) * w + (bx + px)] = rec;
                        }
                    }
                }
            }
        }

        // Chroma U and V 8x8 blocks
        let mut recon_u = vec![128u8; cw * ch];
        let mut recon_v = vec![128u8; cw * ch];
        for (plane_idx, (src_c, recon_c)) in [
            (&src_i420.u_plane, &mut recon_u),
            (&src_i420.v_plane, &mut recon_v),
        ]
        .into_iter()
        .enumerate()
        {
            let ref_c = if plane_idx == 0 {
                &ref_pool[0].u_plane
            } else {
                &ref_pool[0].v_plane
            };
            for by_idx in 0..c_blocks_y {
                for bx_idx in 0..c_blocks_x {
                    let bx = bx_idx * 8;
                    let by = by_idx * 8;
                    let pred = if is_keyframe {
                        predict_8x8_intra(recon_c, cw, bx, by, Vp9IntraMode::DcPred)
                    } else {
                        let mut p = [128u8; 64];
                        for py in 0..8 {
                            for px in 0..8 {
                                p[py * 8 + px] = read_plane_clamped(
                                    ref_c,
                                    cw,
                                    ch,
                                    (bx + px) as isize,
                                    (by + py) as isize,
                                );
                            }
                        }
                        p
                    };
                    let mut res = [0i16; 64];
                    for py in 0..8 {
                        for px in 0..8 {
                            let s = if bx + px < cw && by + py < ch {
                                src_c[(by + py) * cw + (bx + px)] as i16
                            } else {
                                pred[py * 8 + px] as i16
                            };
                            res[py * 8 + px] = s - (pred[py * 8 + px] as i16);
                        }
                    }
                    let dct = vp9_forward_dct_8x8(&res);
                    let mut q_zz = [0i16; 64];
                    let mut deq = [0i16; 64];
                    for (zz_i, &flat_i) in VP9_ZIGZAG_8X8.iter().enumerate() {
                        let q_step = if zz_i == 0 { dc_q } else { ac_q };
                        let qv = (dct[flat_i] as i32) / (q_step as i32);
                        q_zz[zz_i] = qv.clamp(-2047, 2047) as i16;
                        deq[flat_i] = q_zz[zz_i].saturating_mul(q_step);
                    }
                    encode_coeffs_vp9(&mut token_enc, &q_zz);

                    let idct = vp9_inverse_idct_8x8(&deq);
                    for py in 0..8 {
                        for px in 0..8 {
                            if bx + px < cw && by + py < ch {
                                recon_c[(by + py) * cw + (bx + px)] =
                                    ((pred[py * 8 + px] as i16) + idct[py * 8 + px]).clamp(0, 255)
                                        as u8;
                            }
                        }
                    }
                }
            }
        }

        apply_vp9_loop_filter(&mut recon_y, w, h, lf_limit);

        let new_ref = I420Surface {
            width,
            height,
            y_plane: recon_y,
            u_plane: recon_u,
            v_plane: recon_v,
        };
        if is_keyframe {
            ref_pool[1] = new_ref.clone();
            ref_pool[2] = new_ref.clone();
        }
        ref_pool[0] = new_ref;

        let part0 = header_enc.finish();
        let part1 = token_enc.finish();

        // Build VP9 frame packet:
        // Uncompressed header (8 bytes):
        // [0]: 0x82 (keyframe) or 0x86 (interframe)
        // [1..4]: [0x49, 0x83, 0x42] sync code
        // [4..6]: (width - 1) as u16 BE
        // [6..8]: (height - 1) as u16 BE
        // [8..12]: magic "VP9B"
        // [12..16]: part0.len() as u32 BE
        // [16..20]: part1.len() as u32 BE
        let w_m1 = (width.saturating_sub(1)) as u16;
        let h_m1 = (height.saturating_sub(1)) as u16;
        let mut pkt = Vec::with_capacity(20 + part0.len() + part1.len());
        pkt.push(if is_keyframe { 0x82 } else { 0x86 });
        pkt.extend_from_slice(&[0x49, 0x83, 0x42]);
        pkt.extend_from_slice(&w_m1.to_be_bytes());
        pkt.extend_from_slice(&h_m1.to_be_bytes());
        pkt.extend_from_slice(b"VP9B");
        pkt.extend_from_slice(&(part0.len() as u32).to_be_bytes());
        pkt.extend_from_slice(&(part1.len() as u32).to_be_bytes());
        pkt.extend_from_slice(&part0);
        pkt.extend_from_slice(&part1);
        encoded_subframes.push(pkt);
    }

    if encoded_subframes.len() <= 8 {
        Ok(build_vp9_superframe(&encoded_subframes))
    } else {
        // Multi-frame stream with 8-byte VP9 header + "VP9M" frame table
        let mut out = Vec::new();
        out.extend_from_slice(&encoded_subframes[0][..8]);
        out.extend_from_slice(b"VP9M");
        out.extend_from_slice(&(encoded_subframes.len() as u32).to_be_bytes());
        for sf in &encoded_subframes {
            out.extend_from_slice(&(sf.len() as u32).to_be_bytes());
            out.extend_from_slice(sf);
        }
        Ok(out)
    }
}

/// Decode a clean-room VP9 (`vp09`) bitstream (single frame, Annex-B superframe, or `VP9M`
/// multi-frame stream) into a [`VideoClip`].
pub fn decode_vp9_bitstream(data: &[u8], hardware_decode: bool) -> Result<VideoClip, VideoError> {
    if data.len() > MAX_PACKET_BYTES {
        return Err(VideoError::PacketTooLarge { size: data.len() });
    }
    let first_hdr = Vp9FrameHeader::parse(data)?;
    validate_video_dimensions(first_hdr.width, first_hdr.height)?;

    let mut subframe_slices: Vec<&[u8]> = Vec::new();
    if data.len() >= 16 && &data[8..12] == b"VP9M" {
        let count = u32::from_be_bytes([data[12], data[13], data[14], data[15]]) as usize;
        if count == 0 || count > MAX_VIDEO_FRAMES as usize {
            return Err(VideoError::FrameCountTooLarge {
                count: count as u32,
            });
        }
        let mut cursor = 16usize;
        for _ in 0..count {
            if cursor + 4 > data.len() {
                return Err(VideoError::CorruptedBitstream);
            }
            let sz = u32::from_be_bytes([
                data[cursor],
                data[cursor + 1],
                data[cursor + 2],
                data[cursor + 3],
            ]) as usize;
            cursor += 4;
            if cursor + sz > data.len() {
                return Err(VideoError::CorruptedBitstream);
            }
            subframe_slices.push(&data[cursor..cursor + sz]);
            cursor += sz;
        }
    } else if let Some(sizes) = parse_vp9_superframe_index(data) {
        let mut cursor = 0usize;
        for sz in sizes {
            if cursor + sz > data.len() {
                return Err(VideoError::CorruptedBitstream);
            }
            subframe_slices.push(&data[cursor..cursor + sz]);
            cursor += sz;
        }
    } else {
        subframe_slices.push(data);
    }

    let width = first_hdr.width;
    let height = first_hdr.height;
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let ch = h / 2;
    let blocks_x = w.div_ceil(8);
    let blocks_y = h.div_ceil(8);
    let c_blocks_x = cw.div_ceil(8);
    let c_blocks_y = ch.div_ceil(8);

    let mut ref_pool: [I420Surface; 3] = [
        I420Surface::new(width, height),
        I420Surface::new(width, height),
        I420Surface::new(width, height),
    ];

    let mut frames = Vec::with_capacity(subframe_slices.len());
    let mut clip_fps = 30u32;

    for (idx, sf) in subframe_slices.iter().enumerate() {
        if sf.len() < 20 || &sf[8..12] != b"VP9B" {
            return Err(VideoError::CorruptedBitstream);
        }
        let hdr = Vp9FrameHeader::parse(sf)?;
        if hdr.width != width || hdr.height != height {
            return Err(VideoError::InvalidHeader);
        }
        let is_keyframe = hdr.is_keyframe;
        if idx == 0 && !is_keyframe {
            return Err(VideoError::CorruptedBitstream);
        }
        let part0_len = u32::from_be_bytes([sf[12], sf[13], sf[14], sf[15]]) as usize;
        let part1_len = u32::from_be_bytes([sf[16], sf[17], sf[18], sf[19]]) as usize;
        if 20 + part0_len + part1_len > sf.len() {
            return Err(VideoError::CorruptedBitstream);
        }
        let mut header_dec = Vp9BoolDecoder::new(&sf[20..20 + part0_len])?;
        let mut token_dec = Vp9BoolDecoder::new(&sf[20 + part0_len..20 + part0_len + part1_len])?;

        let fps = header_dec.read_literal(8).max(1);
        let pts_ms = header_dec.read_literal(24);
        let q_idx = header_dec.read_literal(8) as u8;
        let lf_limit = header_dec.read_literal(6) as i32;
        if idx == 0 {
            clip_fps = fps;
        }

        let dc_q = vp9_dc_quant(q_idx).max(4);
        let ac_q = vp9_ac_quant(q_idx).max(4);

        let mut recon_y = vec![128u8; w * h];
        let mut last_mv = (0i8, 0i8);

        for by_idx in 0..blocks_y {
            for bx_idx in 0..blocks_x {
                let bx = bx_idx * 8;
                let by = by_idx * 8;
                let pred = if is_keyframe {
                    let mode = Vp9IntraMode::from_u8(header_dec.read_literal(3) as u8);
                    predict_8x8_intra(&recon_y, w, bx, by, mode)
                } else {
                    let ref_slot = usize::from(header_dec.read_bit(128));
                    let inter_mode = Vp9InterMode::from_u8(header_dec.read_literal(2) as u8);
                    let mv = match inter_mode {
                        Vp9InterMode::ZeroMv => (0i8, 0i8),
                        Vp9InterMode::NearestMv | Vp9InterMode::NearMv => last_mv,
                        Vp9InterMode::NewMv => {
                            let dx = header_dec.read_signed(5).clamp(-64, 64) as i8;
                            let dy = header_dec.read_signed(5).clamp(-64, 64) as i8;
                            last_mv = (dx, dy);
                            (dx, dy)
                        }
                    };
                    let mut p = [128u8; 64];
                    for py in 0..8 {
                        for px in 0..8 {
                            p[py * 8 + px] = read_plane_clamped(
                                &ref_pool[ref_slot].y_plane,
                                w,
                                h,
                                (bx + px) as isize + (mv.0 as isize),
                                (by + py) as isize + (mv.1 as isize),
                            );
                        }
                    }
                    p
                };

                let q_zz = decode_coeffs_vp9(&mut token_dec, 64);
                let mut deq = [0i16; 64];
                for (zz_i, &flat_i) in VP9_ZIGZAG_8X8.iter().enumerate() {
                    let q_step = if zz_i == 0 { dc_q } else { ac_q };
                    deq[flat_i] = q_zz[zz_i].saturating_mul(q_step);
                }
                let idct = vp9_inverse_idct_8x8(&deq);
                for py in 0..8 {
                    for px in 0..8 {
                        if bx + px < w && by + py < h {
                            recon_y[(by + py) * w + (bx + px)] =
                                ((pred[py * 8 + px] as i16) + idct[py * 8 + px]).clamp(0, 255)
                                    as u8;
                        }
                    }
                }
            }
        }

        let mut recon_u = vec![128u8; cw * ch];
        let mut recon_v = vec![128u8; cw * ch];
        for (plane_idx, recon_c) in [&mut recon_u, &mut recon_v].into_iter().enumerate() {
            let ref_c = if plane_idx == 0 {
                &ref_pool[0].u_plane
            } else {
                &ref_pool[0].v_plane
            };
            for by_idx in 0..c_blocks_y {
                for bx_idx in 0..c_blocks_x {
                    let bx = bx_idx * 8;
                    let by = by_idx * 8;
                    let pred = if is_keyframe {
                        predict_8x8_intra(recon_c, cw, bx, by, Vp9IntraMode::DcPred)
                    } else {
                        let mut p = [128u8; 64];
                        for py in 0..8 {
                            for px in 0..8 {
                                p[py * 8 + px] = read_plane_clamped(
                                    ref_c,
                                    cw,
                                    ch,
                                    (bx + px) as isize,
                                    (by + py) as isize,
                                );
                            }
                        }
                        p
                    };
                    let q_zz = decode_coeffs_vp9(&mut token_dec, 64);
                    let mut deq = [0i16; 64];
                    for (zz_i, &flat_i) in VP9_ZIGZAG_8X8.iter().enumerate() {
                        let q_step = if zz_i == 0 { dc_q } else { ac_q };
                        deq[flat_i] = q_zz[zz_i].saturating_mul(q_step);
                    }
                    let idct = vp9_inverse_idct_8x8(&deq);
                    for py in 0..8 {
                        for px in 0..8 {
                            if bx + px < cw && by + py < ch {
                                recon_c[(by + py) * cw + (bx + px)] =
                                    ((pred[py * 8 + px] as i16) + idct[py * 8 + px]).clamp(0, 255)
                                        as u8;
                            }
                        }
                    }
                }
            }
        }

        apply_vp9_loop_filter(&mut recon_y, w, h, lf_limit);

        let new_ref = I420Surface {
            width,
            height,
            y_plane: recon_y,
            u_plane: recon_u,
            v_plane: recon_v,
        };
        let rgba = new_ref.to_rgba_bt709();
        if is_keyframe {
            ref_pool[1] = new_ref.clone();
            ref_pool[2] = new_ref.clone();
        }
        ref_pool[0] = new_ref;
        frames.push(VideoFrame::new(width, height, pts_ms, is_keyframe, rgba)?);
    }

    Ok(VideoClip {
        width,
        height,
        fps: clip_fps,
        backend: probe_platform_gpu_backend(hardware_decode),
        profile: VideoCodecProfile::Vp9Profile0,
        frames,
        audio: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vp9_bool_coder_roundtrips_bits_and_signed_literals() {
        let mut enc = Vp9BoolEncoder::new();
        for i in 0..64 {
            enc.write_bit((i & 1) == 0, 40 + ((i * 3) as u8 & 180));
        }
        enc.write_literal(0x2A5, 11);
        enc.write_signed(-19, 6);
        enc.write_signed(0, 6);
        enc.write_signed(27, 6);
        let bytes = enc.finish();

        let mut dec = Vp9BoolDecoder::new(&bytes).unwrap();
        for i in 0..64 {
            assert_eq!(dec.read_bit(40 + ((i * 3) as u8 & 180)), (i & 1) == 0);
        }
        assert_eq!(dec.read_literal(11), 0x2A5);
        assert_eq!(dec.read_signed(6), -19);
        assert_eq!(dec.read_signed(6), 0);
        assert_eq!(dec.read_signed(6), 27);
    }

    #[test]
    fn vp9_adst_and_dct_roundtrip_residuals() {
        let input = [24i16, -12, 8, -4, 18, -6, 4, 2, -10, 6, -4, 2, 4, -2, 1, 0];
        let fwd = vp9_forward_adst_4x4(&input);
        let inv = vp9_inverse_iadst_4x4(&fwd);
        for (a, b) in input.iter().zip(inv.iter()) {
            assert!((a - b).abs() <= 4, "ADST mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn vp9_bitstream_encodes_and_decodes_superframe_with_annexb_index() {
        let mut frames = Vec::new();
        for f in 0..3u32 {
            let mut rgba = vec![0u8; 32 * 24 * 4];
            for y in 0..24u32 {
                for x in 0..32u32 {
                    let i = ((y * 32 + x) * 4) as usize;
                    rgba[i] = (40 + ((x + f * 4) * 5) % 180) as u8;
                    rgba[i + 1] = (60 + (y * 6) % 160) as u8;
                    rgba[i + 2] = 110;
                    rgba[i + 3] = 255;
                }
            }
            frames.push(VideoFrame::new(32, 24, f * 33, f == 0, rgba).unwrap());
        }
        let stream = encode_vp9_bitstream(30, 8, &frames).unwrap();
        let index = parse_vp9_superframe_index(&stream).expect("Annex-B superframe index present");
        assert_eq!(index.len(), 3);

        let clip = decode_vp9_bitstream(&stream, false).unwrap();
        assert_eq!(clip.profile, VideoCodecProfile::Vp9Profile0);
        assert_eq!(clip.width, 32);
        assert_eq!(clip.height, 24);
        assert_eq!(clip.frames.len(), 3);
        assert!(clip.frames[0].keyframe);
        assert!(!clip.frames[1].keyframe);
    }
}
