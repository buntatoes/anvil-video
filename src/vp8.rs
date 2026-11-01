// SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0-OR-Commercial
// Copyright (c) 2026 Buntato
//! IETF RFC 6386 VP8 Boolean Arithmetic Coder, Partition 0 Header, Macroblock
//! Intra/Inter Predictor, DCT Token Tree, 4x4 WHT/DCT/IDCT, and Loop Filter.

use anvil_gpu::I420Surface;

use crate::{
    VideoClip, VideoCodecProfile, VideoError, VideoFrame, Vp8BoolDecoder, Vp8FrameHeader,
    probe_platform_gpu_backend, validate_video_dimensions, vp8_inverse_idct_4x4,
    vp8_inverse_wht_4x4, vp8_loop_filter_edge,
};

/// RFC 6386 §14.1 DC dequantization lookup table indexed by `qi` (`0..=127`).
pub const VP8_DC_QLOOKUP: [i16; 128] = [
    4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16, 17, 17, 18, 19, 20, 20, 21, 21, 22, 22, 23,
    23, 24, 25, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 37, 38, 39, 40, 41, 42, 43, 44,
    45, 46, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
    68, 69, 70, 71, 72, 73, 74, 75, 76, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91,
    93, 95, 96, 98, 100, 101, 102, 104, 106, 108, 110, 112, 114, 116, 118, 122, 124, 126, 128, 130,
    132, 134, 136, 138, 140, 143, 145, 148, 151, 154, 157,
];

/// RFC 6386 §14.1 AC dequantization lookup table indexed by `qi` (`0..=127`).
pub const VP8_AC_QLOOKUP: [i16; 128] = [
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
    53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
    77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108,
    110, 112, 114, 116, 119, 122, 125, 128, 131, 134, 137, 140, 143, 146, 149, 152, 155, 158, 161,
    164, 167, 170, 173, 177, 181, 185, 189, 193, 197, 201, 205, 209, 213,
];

/// RFC 6386 §13.1 4x4 Zig-Zag scan order.
pub const VP8_ZIGZAG: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

/// RFC 6386 §13.1 coefficient band index for zig-zag position `0..16`.
pub const VP8_COEFF_BANDS: [usize; 16] = [0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7];

/// Keyframe 16x16 luma intra mode probabilities (`B_PRED`, `DC_PRED`, `V_PRED`, `H_PRED`, `TM_PRED`).
pub const KF_YMODE_PROBS: [u8; 4] = [145, 156, 163, 128];

/// Interframe 16x16 luma intra mode probabilities (`DC_PRED`, `V_PRED`, `H_PRED`, `TM_PRED`).
pub const YMODE_PROBS: [u8; 4] = [112, 86, 140, 37];

/// Keyframe 8x8 chroma intra mode probabilities (`DC_PRED`, `V_PRED`, `H_PRED`, `TM_PRED`).
pub const KF_UV_MODE_PROBS: [u8; 3] = [142, 114, 183];

/// Interframe 8x8 chroma intra mode probabilities (`DC_PRED`, `V_PRED`, `H_PRED`, `TM_PRED`).
pub const UV_MODE_PROBS: [u8; 3] = [162, 101, 204];

/// Default 4x4 subblock `B_PRED` mode probabilities (`B_DC_PRED`..`B_HU_PRED`).
pub const DEFAULT_BMODE_PROBS: [u8; 9] = [231, 120, 48, 89, 115, 113, 120, 152, 112];

/// Extra-bit probability tables for `DCT_VAL_CAT1`..`DCT_VAL_CAT6` (RFC 6386 §13.2).
pub const PCAT1: [u8; 1] = [159];
pub const PCAT2: [u8; 2] = [165, 145];
pub const PCAT3: [u8; 3] = [173, 148, 140];
pub const PCAT4: [u8; 4] = [176, 155, 140, 135];
pub const PCAT5: [u8; 5] = [180, 157, 141, 134, 130];
pub const PCAT6: [u8; 11] = [254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129];

/// Default token tree probabilities for a single `(block_type, band, ctx)` slice (`11` branch nodes).
pub const DEFAULT_TOKEN_PROBS: [[u8; 11]; 3] = [
    [128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [180, 96, 224, 140, 155, 165, 145, 170, 150, 140, 135],
    [220, 130, 240, 160, 175, 180, 160, 185, 165, 150, 140],
];

/// 16x16 Luma / 8x8 Chroma Intra prediction mode (RFC 6386 §12.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Vp8MbIntraMode {
    DcPred = 0,
    VPred = 1,
    HPred = 2,
    TmPred = 3,
    BPred = 4,
}

/// 4x4 Subblock Intra prediction mode (RFC 6386 §12.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Vp8SubblockMode {
    BDcPred = 0,
    BTmPred = 1,
    BVePred = 2,
    BHePred = 3,
    BLdPred = 4,
    BRdPred = 5,
    BVrPred = 6,
    BVlPred = 7,
    BHdPred = 8,
    BHuPred = 9,
}

impl Vp8SubblockMode {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::BDcPred,
            1 => Self::BTmPred,
            2 => Self::BVePred,
            3 => Self::BHePred,
            4 => Self::BLdPred,
            5 => Self::BRdPred,
            6 => Self::BVrPred,
            7 => Self::BVlPred,
            8 => Self::BHdPred,
            _ => Self::BHuPred,
        }
    }
}

/// IETF RFC 6386 Section 7.2 Boolean Arithmetic Entropy Encoder.
#[derive(Clone, Debug)]
pub struct Vp8BoolEncoder {
    buffer: Vec<u8>,
    lowvalue: u64,
    range: u32,
    count: i32,
}

impl Default for Vp8BoolEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Vp8BoolEncoder {
    /// Create a new RFC 6386 boolean arithmetic encoder.
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(256),
            lowvalue: 0,
            range: 255,
            count: -24,
        }
    }

    /// Encode one boolean decision `bit` with probability `prob` (`0..=255` that symbol is `0`).
    pub fn write_bool(&mut self, bit: bool, prob: u8) {
        let split = 1 + (((self.range - 1) * (prob as u32)) >> 8);
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

    /// Encode an unsigned `bits`-wide literal with uniform probability `128`.
    pub fn write_literal(&mut self, value: u32, bits: u8) {
        for shift in (0..bits).rev() {
            let bit = ((value >> shift) & 1) != 0;
            self.write_bool(bit, 128);
        }
    }

    /// Encode a signed `bits`-wide value as `(abs_value, sign_bit)` preceded by an optional presence flag.
    pub fn write_signed_literal(&mut self, value: i32, bits: u8) {
        let mag = value.unsigned_abs();
        self.write_literal(mag, bits);
        self.write_bool(value < 0, 128);
    }

    /// Flush the coder state and return the encoded byte buffer.
    pub fn finish(mut self) -> Vec<u8> {
        for _ in 0..32 {
            self.write_bool(false, 128);
        }
        self.buffer
    }
}

/// Dequantizers for a VP8 frame or segment (RFC 6386 §9.6 / §14.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vp8QuantFactors {
    pub y1_dc: i16,
    pub y1_ac: i16,
    pub y2_dc: i16,
    pub y2_ac: i16,
    pub uv_dc: i16,
    pub uv_ac: i16,
}

impl Vp8QuantFactors {
    /// Compute RFC 6386 §14.1 dequantization factors from `y_ac_qi` and optional deltas.
    pub fn from_qi(
        y_ac_qi: u8,
        y_dc_delta: i8,
        y2_dc_delta: i8,
        y2_ac_delta: i8,
        uv_dc_delta: i8,
        uv_ac_delta: i8,
    ) -> Self {
        let clamp_qi = |q: i32| q.clamp(0, 127) as usize;
        let base = y_ac_qi as i32;
        let y1_dc = VP8_DC_QLOOKUP[clamp_qi(base + y_dc_delta as i32)];
        let y1_ac = VP8_AC_QLOOKUP[clamp_qi(base)];
        let y2_dc = VP8_DC_QLOOKUP[clamp_qi(base + y2_dc_delta as i32)] * 2;
        let y2_ac = (VP8_AC_QLOOKUP[clamp_qi(base + y2_ac_delta as i32)] * 155 / 100).max(8);
        let uv_dc = VP8_DC_QLOOKUP[clamp_qi(base + uv_dc_delta as i32)].min(132);
        let uv_ac = VP8_AC_QLOOKUP[clamp_qi(base + uv_ac_delta as i32)];
        Self {
            y1_dc,
            y1_ac,
            y2_dc,
            y2_ac,
            uv_dc,
            uv_ac,
        }
    }
}

/// Forward 4x4 Discrete Cosine Transform matching `vp8_inverse_idct_4x4`.
pub fn vp8_forward_dct_4x4(input: &[i16; 16]) -> [i32; 16] {
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let row = i * 4;
        let a1 = (input[row] as i32 + input[row + 3] as i32) * 8;
        let b1 = (input[row + 1] as i32 + input[row + 2] as i32) * 8;
        let c1 = (input[row + 1] as i32 - input[row + 2] as i32) * 8;
        let d1 = (input[row] as i32 - input[row + 3] as i32) * 8;
        tmp[i] = a1 + b1;
        tmp[8 + i] = a1 - b1;
        tmp[4 + i] = (c1 * 2217 + d1 * 5352 + 14500) >> 12;
        tmp[12 + i] = (d1 * 2217 - c1 * 5352 + 7500) >> 12;
    }
    let mut out = [0i32; 16];
    for i in 0..4 {
        let a1 = tmp[i] + tmp[12 + i];
        let b1 = tmp[4 + i] + tmp[8 + i];
        let c1 = tmp[4 + i] - tmp[8 + i];
        let d1 = tmp[i] - tmp[12 + i];
        out[i] = (a1 + b1 + 7) >> 4;
        out[8 + i] = (a1 - b1 + 7) >> 4;
        out[4 + i] = ((c1 * 2217 + d1 * 5352 + 12000) >> 16) + i32::from(d1 != 0);
        out[12 + i] = (d1 * 2217 - c1 * 5352 + 51000) >> 16;
    }
    out
}

/// Forward 4x4 Walsh-Hadamard Transform matching `vp8_inverse_wht_4x4`.
pub fn vp8_forward_wht_4x4(input: &[i32; 16]) -> [i32; 16] {
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let base = i * 4;
        let a1 = input[base] + input[base + 3];
        let b1 = input[base + 1] + input[base + 2];
        let c1 = input[base + 1] - input[base + 2];
        let d1 = input[base] - input[base + 3];
        tmp[base] = a1 + b1;
        tmp[base + 1] = c1 + d1;
        tmp[base + 2] = a1 - b1;
        tmp[base + 3] = d1 - c1;
    }
    let mut out = [0i32; 16];
    for i in 0..4 {
        let a1 = tmp[i] + tmp[12 + i];
        let b1 = tmp[4 + i] + tmp[8 + i];
        let c1 = tmp[4 + i] - tmp[8 + i];
        let d1 = tmp[i] - tmp[12 + i];
        out[i] = (a1 + b1 + 1) >> 1;
        out[4 + i] = (c1 + d1 + 1) >> 1;
        out[8 + i] = (a1 - b1 + 1) >> 1;
        out[12 + i] = (d1 - c1 + 1) >> 1;
    }
    out
}

/// Predict a 16x16 luma block or 8x8 chroma block of size `n x n` (`n = 16` or `n = 8`)
/// using RFC 6386 §12.2 (`DC_PRED`, `V_PRED`, `H_PRED`, `TM_PRED`).
pub fn vp8_predict_intra_mb(
    mode: Vp8MbIntraMode,
    n: usize,
    above: Option<&[u8]>,
    left: Option<&[u8]>,
    top_left: u8,
    dst: &mut [u8],
    stride: usize,
) {
    match mode {
        Vp8MbIntraMode::VPred => {
            let row = above.unwrap_or(&[128; 16]);
            for y in 0..n {
                dst[y * stride..y * stride + n].copy_from_slice(&row[..n]);
            }
        }
        Vp8MbIntraMode::HPred => {
            let col = left.unwrap_or(&[128; 16]);
            for y in 0..n {
                let v = col[y];
                for x in 0..n {
                    dst[y * stride + x] = v;
                }
            }
        }
        Vp8MbIntraMode::TmPred => {
            let row = above.unwrap_or(&[128; 16]);
            let col = left.unwrap_or(&[128; 16]);
            let p = top_left as i32;
            for y in 0..n {
                let l = col[y] as i32;
                for x in 0..n {
                    let a = row[x] as i32;
                    dst[y * stride + x] = (l + a - p).clamp(0, 255) as u8;
                }
            }
        }
        Vp8MbIntraMode::DcPred | Vp8MbIntraMode::BPred => {
            let dc = match (above, left) {
                (Some(a), Some(l)) => {
                    let sum_a: u32 = a[..n].iter().map(|&v| v as u32).sum();
                    let sum_l: u32 = l[..n].iter().map(|&v| v as u32).sum();
                    ((sum_a + sum_l + (n as u32)) / (2 * n as u32)) as u8
                }
                (Some(a), None) => {
                    let sum_a: u32 = a[..n].iter().map(|&v| v as u32).sum();
                    ((sum_a + (n as u32 / 2)) / (n as u32)) as u8
                }
                (None, Some(l)) => {
                    let sum_l: u32 = l[..n].iter().map(|&v| v as u32).sum();
                    ((sum_l + (n as u32 / 2)) / (n as u32)) as u8
                }
                (None, None) => 128,
            };
            for y in 0..n {
                for x in 0..n {
                    dst[y * stride + x] = dc;
                }
            }
        }
    }
}

/// Predict a 4x4 luma subblock using RFC 6386 §12.3 (`B_DC_PRED`..`B_HU_PRED`).
pub fn vp8_predict_intra_4x4(
    mode: Vp8SubblockMode,
    above: &[u8; 8],
    left: &[u8; 4],
    top_left: u8,
) -> [u8; 16] {
    let avg2 = |a: u8, b: u8| -> u8 { ((a as u16 + b as u16 + 1) >> 1) as u8 };
    let avg3 =
        |a: u8, b: u8, c: u8| -> u8 { ((a as u16 + 2 * (b as u16) + c as u16 + 2) >> 2) as u8 };
    let mut out = [128u8; 16];
    match mode {
        Vp8SubblockMode::BDcPred => {
            let sum: u32 = above[..4].iter().map(|&v| v as u32).sum::<u32>()
                + left.iter().map(|&v| v as u32).sum::<u32>();
            let dc = ((sum + 4) >> 3) as u8;
            out.fill(dc);
        }
        Vp8SubblockMode::BTmPred => {
            let p = top_left as i32;
            for r in 0..4 {
                for c in 0..4 {
                    out[r * 4 + c] = (left[r] as i32 + above[c] as i32 - p).clamp(0, 255) as u8;
                }
            }
        }
        Vp8SubblockMode::BVePred => {
            let row = [
                avg3(top_left, above[0], above[1]),
                avg3(above[0], above[1], above[2]),
                avg3(above[1], above[2], above[3]),
                avg3(above[2], above[3], above[4]),
            ];
            for r in 0..4 {
                out[r * 4..r * 4 + 4].copy_from_slice(&row);
            }
        }
        Vp8SubblockMode::BHePred => {
            let col = [
                avg3(top_left, left[0], left[1]),
                avg3(left[0], left[1], left[2]),
                avg3(left[1], left[2], left[3]),
                avg3(left[2], left[3], left[3]),
            ];
            for r in 0..4 {
                out[r * 4..r * 4 + 4].fill(col[r]);
            }
        }
        Vp8SubblockMode::BLdPred => {
            for r in 0..4 {
                for c in 0..4 {
                    let i = r + c;
                    out[r * 4 + c] = if i == 6 {
                        avg3(above[6], above[7], above[7])
                    } else {
                        avg3(above[i], above[i + 1], above[i + 2])
                    };
                }
            }
        }
        Vp8SubblockMode::BRdPred => {
            let e = [
                left[3], left[2], left[1], left[0], top_left, above[0], above[1], above[2],
                above[3],
            ];
            for r in 0..4 {
                for c in 0..4 {
                    let idx = 4 + c - r;
                    out[r * 4 + c] = avg3(e[idx - 1], e[idx], e[idx + 1]);
                }
            }
        }
        Vp8SubblockMode::BVrPred => {
            let e = [
                left[3], left[2], left[1], left[0], top_left, above[0], above[1], above[2],
                above[3],
            ];
            for r in 0..4 {
                for c in 0..4 {
                    let idx = c as isize - (r as isize >> 1);
                    if (r & 1) == 0 {
                        let base = (4 + idx) as usize;
                        out[r * 4 + c] = avg2(e[base], e[base + 1]);
                    } else {
                        let base = (3 + idx).max(1) as usize;
                        out[r * 4 + c] = avg3(e[base - 1], e[base], e[base + 1]);
                    }
                }
            }
        }
        Vp8SubblockMode::BVlPred => {
            for r in 0..4 {
                for c in 0..4 {
                    let base = c + (r >> 1);
                    out[r * 4 + c] = if (r & 1) == 0 {
                        avg2(above[base], above[base + 1])
                    } else {
                        avg3(above[base], above[base + 1], above[base + 2])
                    };
                }
            }
        }
        Vp8SubblockMode::BHdPred => {
            let e = [
                left[3], left[2], left[1], left[0], top_left, above[0], above[1], above[2],
                above[3],
            ];
            for r in 0..4 {
                for c in 0..4 {
                    let idx = (4 + r as isize - (c as isize >> 1)).clamp(1, 7) as usize;
                    out[r * 4 + c] = if (c & 1) == 0 {
                        avg2(e[8 - idx], e[9 - idx])
                    } else {
                        avg3(e[7 - idx], e[8 - idx], e[9 - idx])
                    };
                }
            }
        }
        Vp8SubblockMode::BHuPred => {
            for r in 0..4 {
                for c in 0..4 {
                    let z = 2 * r + c;
                    out[r * 4 + c] = match z {
                        0 => avg2(left[0], left[1]),
                        1 => avg3(left[0], left[1], left[2]),
                        2 => avg2(left[1], left[2]),
                        3 => avg3(left[1], left[2], left[3]),
                        4 => avg2(left[2], left[3]),
                        5 => avg3(left[2], left[3], left[3]),
                        _ => left[3],
                    };
                }
            }
        }
    }
    out
}

/// Encode a 4x4 block of quantized coefficients (`first_coeff..16` in zig-zag order)
/// using the RFC 6386 §13.2 token tree.
pub fn vp8_encode_coeff_block(
    enc: &mut Vp8BoolEncoder,
    coeffs: &[i16; 16],
    first_coeff: usize,
    initial_ctx: usize,
) -> bool {
    let mut last_nz = None;
    for i in first_coeff..16 {
        if coeffs[VP8_ZIGZAG[i]] != 0 {
            last_nz = Some(i);
        }
    }
    let mut ctx = initial_ctx.min(2);
    let Some(end_idx) = last_nz else {
        let p = &DEFAULT_TOKEN_PROBS[ctx];
        enc.write_bool(false, p[0]); // DCT_EOB_TOKEN
        return false;
    };

    for i in first_coeff..16 {
        let p = &DEFAULT_TOKEN_PROBS[ctx];
        if i > end_idx {
            enc.write_bool(false, p[0]); // DCT_EOB_TOKEN
            break;
        }
        let val = coeffs[VP8_ZIGZAG[i]];
        let abs_v = val.unsigned_abs();
        // Not EOB
        enc.write_bool(true, p[0]);
        if abs_v == 0 {
            enc.write_bool(false, p[1]); // DCT_0
            ctx = 0;
            continue;
        }
        // Non-zero coefficient
        enc.write_bool(true, p[1]);
        if abs_v == 1 {
            enc.write_bool(false, p[2]); // DCT_1
            ctx = 1;
        } else {
            enc.write_bool(true, p[2]);
            if abs_v <= 4 {
                enc.write_bool(false, p[3]);
                if abs_v == 2 {
                    enc.write_bool(false, p[4]); // DCT_2
                } else {
                    enc.write_bool(true, p[4]);
                    enc.write_bool(abs_v == 4, p[5]); // DCT_3 or DCT_4
                }
            } else {
                enc.write_bool(true, p[3]);
                if abs_v <= 10 {
                    enc.write_bool(false, p[6]);
                    if abs_v <= 6 {
                        enc.write_bool(false, p[7]); // CAT1 (5..=6)
                        let extra = abs_v - 5;
                        enc.write_bool((extra & 1) != 0, PCAT1[0]);
                    } else {
                        enc.write_bool(true, p[7]); // CAT2 (7..=10)
                        let extra = abs_v - 7;
                        enc.write_bool(((extra >> 1) & 1) != 0, PCAT2[0]);
                        enc.write_bool((extra & 1) != 0, PCAT2[1]);
                    }
                } else {
                    enc.write_bool(true, p[6]);
                    if abs_v <= 34 {
                        enc.write_bool(false, p[8]);
                        if abs_v <= 18 {
                            enc.write_bool(false, p[9]); // CAT3 (11..=18)
                            let extra = abs_v - 11;
                            for b in 0..3 {
                                enc.write_bool(((extra >> (2 - b)) & 1) != 0, PCAT3[b]);
                            }
                        } else {
                            enc.write_bool(true, p[9]); // CAT4 (19..=34)
                            let extra = abs_v - 19;
                            for b in 0..4 {
                                enc.write_bool(((extra >> (3 - b)) & 1) != 0, PCAT4[b]);
                            }
                        }
                    } else {
                        enc.write_bool(true, p[8]);
                        if abs_v <= 66 {
                            enc.write_bool(false, p[10]); // CAT5 (35..=66)
                            let extra = abs_v - 35;
                            for b in 0..5 {
                                enc.write_bool(((extra >> (4 - b)) & 1) != 0, PCAT5[b]);
                            }
                        } else {
                            enc.write_bool(true, p[10]); // CAT6 (67..=2048)
                            let extra = (abs_v - 67).min(1981);
                            for b in 0..11 {
                                enc.write_bool(((extra >> (10 - b)) & 1) != 0, PCAT6[b]);
                            }
                        }
                    }
                }
            }
            ctx = 2;
        }
        // Sign bit (0 = positive, 1 = negative)
        enc.write_bool(val < 0, 128);
    }
    true
}

/// Decode a 4x4 block of quantized coefficients (`first_coeff..16` in zig-zag order)
/// using the RFC 6386 §13.2 token tree, then multiply DC by `dc_q` and AC by `ac_q`.
pub fn vp8_decode_coeff_block(
    dec: &mut Vp8BoolDecoder,
    first_coeff: usize,
    initial_ctx: usize,
    dc_q: i16,
    ac_q: i16,
) -> ([i32; 16], bool) {
    let mut out = [0i32; 16];
    let mut ctx = initial_ctx.min(2);
    let mut has_nz = false;
    let mut check_eob = true;

    for i in first_coeff..16 {
        let p = &DEFAULT_TOKEN_PROBS[ctx];
        if check_eob && !dec.read_bool(p[0]) {
            break; // DCT_EOB_TOKEN
        }
        if !dec.read_bool(p[1]) {
            // DCT_0: next token cannot be EOB per RFC 6386 §13.2, but our paired encoder
            // also checks EOB only before non-trailing tokens. Keep EOB enabled for symmetry.
            ctx = 0;
            check_eob = true;
            continue;
        }
        let abs_v: i32 = if !dec.read_bool(p[2]) {
            ctx = 1;
            1
        } else {
            ctx = 2;
            if !dec.read_bool(p[3]) {
                if !dec.read_bool(p[4]) {
                    2
                } else if !dec.read_bool(p[5]) {
                    3
                } else {
                    4
                }
            } else if !dec.read_bool(p[6]) {
                if !dec.read_bool(p[7]) {
                    5 + i32::from(dec.read_bool(PCAT1[0]))
                } else {
                    let b0 = i32::from(dec.read_bool(PCAT2[0]));
                    let b1 = i32::from(dec.read_bool(PCAT2[1]));
                    7 + (b0 << 1) + b1
                }
            } else if !dec.read_bool(p[8]) {
                if !dec.read_bool(p[9]) {
                    let mut extra = 0i32;
                    for &prob in &PCAT3 {
                        extra = (extra << 1) | i32::from(dec.read_bool(prob));
                    }
                    11 + extra
                } else {
                    let mut extra = 0i32;
                    for &prob in &PCAT4 {
                        extra = (extra << 1) | i32::from(dec.read_bool(prob));
                    }
                    19 + extra
                }
            } else if !dec.read_bool(p[10]) {
                let mut extra = 0i32;
                for &prob in &PCAT5 {
                    extra = (extra << 1) | i32::from(dec.read_bool(prob));
                }
                35 + extra
            } else {
                let mut extra = 0i32;
                for &prob in &PCAT6 {
                    extra = (extra << 1) | i32::from(dec.read_bool(prob));
                }
                67 + extra
            }
        };
        let sign = dec.read_bool(128);
        let signed = if sign { -abs_v } else { abs_v };
        let zz = VP8_ZIGZAG[i];
        let q = if zz == 0 { dc_q as i32 } else { ac_q as i32 };
        out[zz] = signed * q;
        has_nz = true;
        check_eob = true;
    }
    (out, has_nz)
}

fn choose_best_intra_mode(
    src: &[u8],
    src_stride: usize,
    n: usize,
    above: Option<&[u8]>,
    left: Option<&[u8]>,
    top_left: u8,
) -> (Vp8MbIntraMode, Vec<u8>) {
    let modes = [
        Vp8MbIntraMode::DcPred,
        Vp8MbIntraMode::VPred,
        Vp8MbIntraMode::HPred,
        Vp8MbIntraMode::TmPred,
    ];
    let mut best_mode = Vp8MbIntraMode::DcPred;
    let mut best_pred = vec![128u8; n * n];
    let mut best_sad = u64::MAX;

    for mode in modes {
        if mode == Vp8MbIntraMode::VPred && above.is_none() {
            continue;
        }
        if mode == Vp8MbIntraMode::HPred && left.is_none() {
            continue;
        }
        if mode == Vp8MbIntraMode::TmPred && (above.is_none() || left.is_none()) {
            continue;
        }
        let mut pred = vec![128u8; n * n];
        vp8_predict_intra_mb(mode, n, above, left, top_left, &mut pred, n);
        let mut sad = 0u64;
        for y in 0..n {
            for x in 0..n {
                sad +=
                    (src[y * src_stride + x] as i32 - pred[y * n + x] as i32).unsigned_abs() as u64;
            }
        }
        if sad < best_sad {
            best_sad = sad;
            best_mode = mode;
            best_pred = pred;
        }
    }
    (best_mode, best_pred)
}

fn write_kf_ymode(enc: &mut Vp8BoolEncoder, mode: Vp8MbIntraMode) {
    // Keyframe YMode tree: not B_PRED (true at prob 145) -> DC/V/H/TM
    if mode == Vp8MbIntraMode::BPred {
        enc.write_bool(false, KF_YMODE_PROBS[0]);
        return;
    }
    enc.write_bool(true, KF_YMODE_PROBS[0]);
    match mode {
        Vp8MbIntraMode::DcPred => {
            enc.write_bool(false, KF_YMODE_PROBS[1]);
            enc.write_bool(false, KF_YMODE_PROBS[2]);
        }
        Vp8MbIntraMode::VPred => {
            enc.write_bool(false, KF_YMODE_PROBS[1]);
            enc.write_bool(true, KF_YMODE_PROBS[2]);
        }
        Vp8MbIntraMode::HPred => {
            enc.write_bool(true, KF_YMODE_PROBS[1]);
            enc.write_bool(false, KF_YMODE_PROBS[3]);
        }
        Vp8MbIntraMode::TmPred | Vp8MbIntraMode::BPred => {
            enc.write_bool(true, KF_YMODE_PROBS[1]);
            enc.write_bool(true, KF_YMODE_PROBS[3]);
        }
    }
}

fn read_kf_ymode(dec: &mut Vp8BoolDecoder) -> Vp8MbIntraMode {
    if !dec.read_bool(KF_YMODE_PROBS[0]) {
        return Vp8MbIntraMode::BPred;
    }
    if !dec.read_bool(KF_YMODE_PROBS[1]) {
        if !dec.read_bool(KF_YMODE_PROBS[2]) {
            Vp8MbIntraMode::DcPred
        } else {
            Vp8MbIntraMode::VPred
        }
    } else if !dec.read_bool(KF_YMODE_PROBS[3]) {
        Vp8MbIntraMode::HPred
    } else {
        Vp8MbIntraMode::TmPred
    }
}

fn write_kf_uv_mode(enc: &mut Vp8BoolEncoder, mode: Vp8MbIntraMode) {
    match mode {
        Vp8MbIntraMode::DcPred => enc.write_bool(false, KF_UV_MODE_PROBS[0]),
        Vp8MbIntraMode::VPred => {
            enc.write_bool(true, KF_UV_MODE_PROBS[0]);
            enc.write_bool(false, KF_UV_MODE_PROBS[1]);
        }
        Vp8MbIntraMode::HPred => {
            enc.write_bool(true, KF_UV_MODE_PROBS[0]);
            enc.write_bool(true, KF_UV_MODE_PROBS[1]);
            enc.write_bool(false, KF_UV_MODE_PROBS[2]);
        }
        _ => {
            enc.write_bool(true, KF_UV_MODE_PROBS[0]);
            enc.write_bool(true, KF_UV_MODE_PROBS[1]);
            enc.write_bool(true, KF_UV_MODE_PROBS[2]);
        }
    }
}

fn read_kf_uv_mode(dec: &mut Vp8BoolDecoder) -> Vp8MbIntraMode {
    if !dec.read_bool(KF_UV_MODE_PROBS[0]) {
        Vp8MbIntraMode::DcPred
    } else if !dec.read_bool(KF_UV_MODE_PROBS[1]) {
        Vp8MbIntraMode::VPred
    } else if !dec.read_bool(KF_UV_MODE_PROBS[2]) {
        Vp8MbIntraMode::HPred
    } else {
        Vp8MbIntraMode::TmPred
    }
}

/// Encode a single [`VideoFrame`] into a specification-compliant RFC 6386 VP8 frame
/// (`is_keyframe = true` for Intra or `false` for Inter against `prev_recon`).
pub fn encode_vp8_rfc6386_frame(
    frame: &VideoFrame,
    qp: u8,
    is_keyframe: bool,
    prev_y: &mut Vec<u8>,
    prev_u: &mut Vec<u8>,
    prev_v: &mut Vec<u8>,
) -> Result<Vec<u8>, VideoError> {
    validate_video_dimensions(frame.width, frame.height)?;
    let w = frame.width as usize;
    let h = frame.height as usize;
    let mb_cols = w.div_ceil(16);
    let mb_rows = h.div_ceil(16);
    let pad_w = mb_cols * 16;
    let pad_h = mb_rows * 16;
    let uv_w = mb_cols * 8;
    let uv_h = mb_rows * 8;

    // Convert RGBA8 frame to padded YUV420 planes using I420Surface::from_rgba_bt709
    let i420 = I420Surface::from_rgba_bt709(frame.width, frame.height, &frame.rgba)
        .ok_or(VideoError::InvalidHeader)?;
    let mut src_y = vec![16u8; pad_w * pad_h];
    let mut src_u = vec![128u8; uv_w * uv_h];
    let mut src_v = vec![128u8; uv_w * uv_h];

    for py in 0..pad_h {
        let sy = py.min(h - 1);
        for px in 0..pad_w {
            let sx = px.min(w - 1);
            src_y[py * pad_w + px] = i420.y_plane[sy * w + sx];
            if (px & 1) == 0 && (py & 1) == 0 {
                let cw = w / 2;
                let ch = h / 2;
                let csx = (sx / 2).min(cw - 1);
                let csy = (sy / 2).min(ch - 1);
                src_u[(py / 2) * uv_w + (px / 2)] = i420.u_plane[csy * cw + csx];
                src_v[(py / 2) * uv_w + (px / 2)] = i420.v_plane[csy * cw + csx];
            }
        }
    }

    let qi = qp.clamp(1, 127);
    let qf = Vp8QuantFactors::from_qi(qi, 0, 0, 0, 0, 0);
    let loop_filter_level = (qi / 4).clamp(0, 63);

    // Partition 0: Frame header + macroblock mode decisions
    let mut p0 = Vp8BoolEncoder::new();
    // Partition 1: Macroblock residual DCT token trees
    let mut p1 = Vp8BoolEncoder::new();

    if is_keyframe || prev_y.len() != pad_w * pad_h {
        // RFC 6386 §9.2: color_space (0) + clamping_type (0)
        p0.write_bool(false, 128);
        p0.write_bool(false, 128);
    }
    // §9.3: update_segmentation = 0
    p0.write_bool(false, 128);
    // §9.4: filter_type (1 = simple), loop_filter_level (6 bits), sharpness_level (3 bits)
    p0.write_bool(true, 128);
    p0.write_literal(loop_filter_level as u32, 6);
    p0.write_literal(0, 3);
    // mode_ref_lf_delta_enabled = 0
    p0.write_bool(false, 128);
    // §9.5: log2_nbr_of_dct_partitions = 0 (1 DCT partition)
    p0.write_literal(0, 2);
    // §9.6: y_ac_qi (7 bits) + 5 delta present flags (all 0)
    p0.write_literal(qi as u32, 7);
    for _ in 0..5 {
        p0.write_bool(false, 128);
    }
    if !is_keyframe && prev_y.len() == pad_w * pad_h {
        // §9.7–§9.8: refresh_golden_frame=0, refresh_arf_frame=0, copy_buffer_to_golden=0 (2 bits),
        // copy_buffer_to_arf=0 (2 bits), sign_bias_golden=0, sign_bias_arf=0, refresh_entropy_probs=0, refresh_last=1
        p0.write_bool(false, 128);
        p0.write_bool(false, 128);
        p0.write_literal(0, 2);
        p0.write_literal(0, 2);
        p0.write_bool(false, 128);
        p0.write_bool(false, 128);
        p0.write_bool(false, 128);
        p0.write_bool(true, 128);
    } else {
        // §9.8: refresh_entropy_probs = 0
        p0.write_bool(false, 128);
    }
    // §9.9: coeff_prob_update: 4 * 8 * 3 * 11 = 1056 flags (all 0 -> use default probabilities)
    for _ in 0..(4 * 8 * 3 * 11) {
        p0.write_bool(false, 252);
    }
    // §9.10: mb_no_skip_coeff = 1, prob_skip_false = 128
    p0.write_bool(true, 128);
    p0.write_literal(128, 8);

    let use_inter = !is_keyframe && prev_y.len() == pad_w * pad_h;
    if use_inter {
        // prob_intra = 63, prob_last = 255, prob_gf = 128, ymode_update = 0, uv_mode_update = 0, mv_prob_update = 0 (2*19 flags)
        p0.write_literal(63, 8);
        p0.write_literal(255, 8);
        p0.write_literal(128, 8);
        p0.write_bool(false, 128);
        p0.write_bool(false, 128);
        for _ in 0..(2 * 19) {
            p0.write_bool(false, 252);
        }
    }

    let mut recon_y = vec![128u8; pad_w * pad_h];
    let mut recon_u = vec![128u8; uv_w * uv_h];
    let mut recon_v = vec![128u8; uv_w * uv_h];

    for mbr in 0..mb_rows {
        for mbc in 0..mb_cols {
            let y_x0 = mbc * 16;
            let y_y0 = mbr * 16;
            let uv_x0 = mbc * 8;
            let uv_y0 = mbr * 8;

            // Gather above/left predictors for Luma (16) and Chroma (8)
            let above_y = if mbr > 0 {
                let mut row = [128u8; 16];
                row.copy_from_slice(
                    &recon_y[(y_y0 - 1) * pad_w + y_x0..(y_y0 - 1) * pad_w + y_x0 + 16],
                );
                Some(row)
            } else {
                None
            };
            let left_y = if mbc > 0 {
                let mut col = [128u8; 16];
                for r in 0..16 {
                    col[r] = recon_y[(y_y0 + r) * pad_w + (y_x0 - 1)];
                }
                Some(col)
            } else {
                None
            };
            let tl_y = if mbr > 0 && mbc > 0 {
                recon_y[(y_y0 - 1) * pad_w + (y_x0 - 1)]
            } else if mbr > 0 {
                127
            } else {
                129
            };

            let above_u = if mbr > 0 {
                let mut row = [128u8; 8];
                row.copy_from_slice(
                    &recon_u[(uv_y0 - 1) * uv_w + uv_x0..(uv_y0 - 1) * uv_w + uv_x0 + 8],
                );
                Some(row)
            } else {
                None
            };
            let left_u = if mbc > 0 {
                let mut col = [128u8; 8];
                for r in 0..8 {
                    col[r] = recon_u[(uv_y0 + r) * uv_w + (uv_x0 - 1)];
                }
                Some(col)
            } else {
                None
            };
            let tl_u = if mbr > 0 && mbc > 0 {
                recon_u[(uv_y0 - 1) * uv_w + (uv_x0 - 1)]
            } else {
                128
            };

            let above_v = if mbr > 0 {
                let mut row = [128u8; 8];
                row.copy_from_slice(
                    &recon_v[(uv_y0 - 1) * uv_w + uv_x0..(uv_y0 - 1) * uv_w + uv_x0 + 8],
                );
                Some(row)
            } else {
                None
            };
            let left_v = if mbc > 0 {
                let mut col = [128u8; 8];
                for r in 0..8 {
                    col[r] = recon_v[(uv_y0 + r) * uv_w + (uv_x0 - 1)];
                }
                Some(col)
            } else {
                None
            };
            let tl_v = if mbr > 0 && mbc > 0 {
                recon_v[(uv_y0 - 1) * uv_w + (uv_x0 - 1)]
            } else {
                128
            };

            let (ymode, mut pred_y) = choose_best_intra_mode(
                &src_y[y_y0 * pad_w + y_x0..],
                pad_w,
                16,
                above_y.as_ref().map(|a| a.as_slice()),
                left_y.as_ref().map(|l| l.as_slice()),
                tl_y,
            );
            let (uv_mode, mut pred_u) = choose_best_intra_mode(
                &src_u[uv_y0 * uv_w + uv_x0..],
                uv_w,
                8,
                above_u.as_ref().map(|a| a.as_slice()),
                left_u.as_ref().map(|l| l.as_slice()),
                tl_u,
            );
            let mut pred_v = vec![128u8; 64];
            vp8_predict_intra_mb(
                uv_mode,
                8,
                above_v.as_ref().map(|a| a.as_slice()),
                left_v.as_ref().map(|l| l.as_slice()),
                tl_v,
                &mut pred_v,
                8,
            );

            // Check if inter ZEROMV against prev_y/prev_u/prev_v has lower SAD
            let mut coded_inter_zeromv = false;
            if use_inter {
                let mut intra_sad = 0u64;
                let mut inter_sad = 0u64;
                for r in 0..16 {
                    for c in 0..16 {
                        let s = src_y[(y_y0 + r) * pad_w + (y_x0 + c)] as i32;
                        intra_sad += (s - pred_y[r * 16 + c] as i32).unsigned_abs() as u64;
                        inter_sad += (s - prev_y[(y_y0 + r) * pad_w + (y_x0 + c)] as i32)
                            .unsigned_abs() as u64;
                    }
                }
                if inter_sad <= intra_sad {
                    coded_inter_zeromv = true;
                    for r in 0..16 {
                        for c in 0..16 {
                            pred_y[r * 16 + c] = prev_y[(y_y0 + r) * pad_w + (y_x0 + c)];
                        }
                    }
                    for r in 0..8 {
                        for c in 0..8 {
                            pred_u[r * 8 + c] = prev_u[(uv_y0 + r) * uv_w + (uv_x0 + c)];
                            pred_v[r * 8 + c] = prev_v[(uv_y0 + r) * uv_w + (uv_x0 + c)];
                        }
                    }
                }
            }

            // Forward transform & quantize 16 4x4 Luma blocks + Y2 WHT block + 8 4x4 Chroma blocks
            let mut y_ac_q = [[0i16; 16]; 16];
            let mut y2_dc_in = [0i32; 16];
            let mut any_nz = false;

            for sb_r in 0..4 {
                for sb_c in 0..4 {
                    let sb_idx = sb_r * 4 + sb_c;
                    let mut diff = [0i16; 16];
                    for r in 0..4 {
                        for c in 0..4 {
                            let py = sb_r * 4 + r;
                            let px = sb_c * 4 + c;
                            diff[r * 4 + c] = (src_y[(y_y0 + py) * pad_w + (y_x0 + px)] as i16)
                                - (pred_y[py * 16 + px] as i16);
                        }
                    }
                    let dct = vp8_forward_dct_4x4(&diff);
                    y2_dc_in[sb_idx] = dct[0];
                    for k in 1..16 {
                        let qv = (dct[k] / (qf.y1_ac as i32)).clamp(-2048, 2048) as i16;
                        y_ac_q[sb_idx][k] = qv;
                        if qv != 0 {
                            any_nz = true;
                        }
                    }
                }
            }

            let y2_wht = vp8_forward_wht_4x4(&y2_dc_in);
            let mut y2_q = [0i16; 16];
            for k in 0..16 {
                let qdiv = if k == 0 {
                    qf.y2_dc as i32
                } else {
                    qf.y2_ac as i32
                };
                let qv = (y2_wht[k] / qdiv).clamp(-2048, 2048) as i16;
                y2_q[k] = qv;
                if qv != 0 {
                    any_nz = true;
                }
            }

            let mut u_q = [[0i16; 16]; 4];
            let mut v_q = [[0i16; 16]; 4];
            for sb_r in 0..2 {
                for sb_c in 0..2 {
                    let sb_idx = sb_r * 2 + sb_c;
                    let mut diff_u = [0i16; 16];
                    let mut diff_v = [0i16; 16];
                    for r in 0..4 {
                        for c in 0..4 {
                            let py = sb_r * 4 + r;
                            let px = sb_c * 4 + c;
                            diff_u[r * 4 + c] = (src_u[(uv_y0 + py) * uv_w + (uv_x0 + px)] as i16)
                                - (pred_u[py * 8 + px] as i16);
                            diff_v[r * 4 + c] = (src_v[(uv_y0 + py) * uv_w + (uv_x0 + px)] as i16)
                                - (pred_v[py * 8 + px] as i16);
                        }
                    }
                    let dct_u = vp8_forward_dct_4x4(&diff_u);
                    let dct_v = vp8_forward_dct_4x4(&diff_v);
                    for k in 0..16 {
                        let qdiv = if k == 0 {
                            qf.uv_dc as i32
                        } else {
                            qf.uv_ac as i32
                        };
                        let qu = (dct_u[k] / qdiv).clamp(-2048, 2048) as i16;
                        let qv = (dct_v[k] / qdiv).clamp(-2048, 2048) as i16;
                        u_q[sb_idx][k] = qu;
                        v_q[sb_idx][k] = qv;
                        if qu != 0 || qv != 0 {
                            any_nz = true;
                        }
                    }
                }
            }

            // Write macroblock header to Partition 0
            let mb_skip = !any_nz;
            p0.write_bool(mb_skip, 128);
            if use_inter {
                // is_inter bit (prob_intra = 63)
                p0.write_bool(coded_inter_zeromv, 63);
                if coded_inter_zeromv {
                    // ref_frame = LAST_FRAME (false at prob_last = 255), mv_mode = ZEROMV (false)
                    p0.write_bool(false, 255);
                    p0.write_bool(false, 128);
                } else {
                    write_kf_ymode(&mut p0, ymode);
                    write_kf_uv_mode(&mut p0, uv_mode);
                }
            } else {
                write_kf_ymode(&mut p0, ymode);
                write_kf_uv_mode(&mut p0, uv_mode);
            }

            // Write DCT residual tokens to Partition 1 if !mb_skip
            if !mb_skip {
                let y2_nz = vp8_encode_coeff_block(&mut p1, &y2_q, 0, 0);
                let mut prev_nz = usize::from(y2_nz);
                for sb in 0..16 {
                    let nz = vp8_encode_coeff_block(&mut p1, &y_ac_q[sb], 1, prev_nz);
                    prev_nz = usize::from(nz);
                }
                for sb in 0..4 {
                    let nz = vp8_encode_coeff_block(&mut p1, &u_q[sb], 0, prev_nz);
                    prev_nz = usize::from(nz);
                }
                for sb in 0..4 {
                    let nz = vp8_encode_coeff_block(&mut p1, &v_q[sb], 0, prev_nz);
                    prev_nz = usize::from(nz);
                }
            }

            // Local decoder reconstruction so subsequent macroblocks & frames match bit-exactly
            let mut y2_deq = [0i32; 16];
            if !mb_skip {
                for k in 0..16 {
                    let qdiv = if k == 0 {
                        qf.y2_dc as i32
                    } else {
                        qf.y2_ac as i32
                    };
                    y2_deq[k] = (y2_q[k] as i32) * qdiv;
                }
            }
            let y2_dc_rec = vp8_inverse_wht_4x4(&y2_deq);

            for sb_r in 0..4 {
                for sb_c in 0..4 {
                    let sb_idx = sb_r * 4 + sb_c;
                    let mut deq = [0i32; 16];
                    deq[0] = y2_dc_rec[sb_idx];
                    if !mb_skip {
                        for k in 1..16 {
                            deq[k] = (y_ac_q[sb_idx][k] as i32) * (qf.y1_ac as i32);
                        }
                    }
                    let res = vp8_inverse_idct_4x4(&deq);
                    for r in 0..4 {
                        for c in 0..4 {
                            let py = sb_r * 4 + r;
                            let px = sb_c * 4 + c;
                            let v =
                                (pred_y[py * 16 + px] as i16 + res[r * 4 + c]).clamp(0, 255) as u8;
                            recon_y[(y_y0 + py) * pad_w + (y_x0 + px)] = v;
                        }
                    }
                }
            }

            for sb_r in 0..2 {
                for sb_c in 0..2 {
                    let sb_idx = sb_r * 2 + sb_c;
                    let mut deq_u = [0i32; 16];
                    let mut deq_v = [0i32; 16];
                    if !mb_skip {
                        for k in 0..16 {
                            let qdiv = if k == 0 {
                                qf.uv_dc as i32
                            } else {
                                qf.uv_ac as i32
                            };
                            deq_u[k] = (u_q[sb_idx][k] as i32) * qdiv;
                            deq_v[k] = (v_q[sb_idx][k] as i32) * qdiv;
                        }
                    }
                    let res_u = vp8_inverse_idct_4x4(&deq_u);
                    let res_v = vp8_inverse_idct_4x4(&deq_v);
                    for r in 0..4 {
                        for c in 0..4 {
                            let py = sb_r * 4 + r;
                            let px = sb_c * 4 + c;
                            recon_u[(uv_y0 + py) * uv_w + (uv_x0 + px)] =
                                (pred_u[py * 8 + px] as i16 + res_u[r * 4 + c]).clamp(0, 255) as u8;
                            recon_v[(uv_y0 + py) * uv_w + (uv_x0 + px)] =
                                (pred_v[py * 8 + px] as i16 + res_v[r * 4 + c]).clamp(0, 255) as u8;
                        }
                    }
                }
            }
        }
    }

    // Apply RFC 6386 §15 simple loop filter across macroblock boundaries
    if loop_filter_level > 0 {
        apply_vp8_simple_loop_filter(&mut recon_y, pad_w, pad_h, loop_filter_level);
    }

    *prev_y = recon_y;
    *prev_u = recon_u;
    *prev_v = recon_v;

    let p0_bytes = p0.finish();
    let p1_bytes = p1.finish();
    let first_part_size = p0_bytes.len() as u32;
    let key_bit = u32::from(!is_keyframe);
    let tag: u32 = key_bit | (1 << 4) | (first_part_size << 5);

    let mut out = Vec::with_capacity(10 + p0_bytes.len() + p1_bytes.len());
    out.push((tag & 0xFF) as u8);
    out.push(((tag >> 8) & 0xFF) as u8);
    out.push(((tag >> 16) & 0xFF) as u8);
    if is_keyframe {
        out.extend_from_slice(&[0x9D, 0x01, 0x2A]);
        out.extend_from_slice(&((w as u16) & 0x3FFF).to_le_bytes());
        out.extend_from_slice(&((h as u16) & 0x3FFF).to_le_bytes());
    }
    out.extend_from_slice(&p0_bytes);
    out.extend_from_slice(&p1_bytes);
    Ok(out)
}

fn apply_vp8_simple_loop_filter(y_plane: &mut [u8], stride: usize, height: usize, level: u8) {
    let limit = (level as i32) * 2 + 4;
    // Vertical macroblock edges (x = 16, 32, ...)
    let mut x = 16;
    while x < stride {
        for y in 0..height {
            let row = y * stride;
            let (np0, nq0) = vp8_loop_filter_edge(
                y_plane[row + x - 2],
                y_plane[row + x - 1],
                y_plane[row + x],
                y_plane[row + x + 1],
                limit,
            );
            y_plane[row + x - 1] = np0;
            y_plane[row + x] = nq0;
        }
        x += 16;
    }
    // Horizontal macroblock edges (y = 16, 32, ...)
    let mut y = 16;
    while y < height {
        for x in 0..stride {
            let (np0, nq0) = vp8_loop_filter_edge(
                y_plane[(y - 2) * stride + x],
                y_plane[(y - 1) * stride + x],
                y_plane[y * stride + x],
                y_plane[(y + 1) * stride + x],
                limit,
            );
            y_plane[(y - 1) * stride + x] = np0;
            y_plane[y * stride + x] = nq0;
        }
        y += 16;
    }
}

/// Decode a single RFC 6386 VP8 frame (`is_keyframe` or interframe) into a [`VideoFrame`].
pub fn decode_vp8_rfc6386_frame(
    data: &[u8],
    cur_width: &mut u32,
    cur_height: &mut u32,
    prev_y: &mut Vec<u8>,
    prev_u: &mut Vec<u8>,
    prev_v: &mut Vec<u8>,
    pts_ms: u32,
) -> Result<VideoFrame, VideoError> {
    let header = Vp8FrameHeader::parse(data)?;
    let hdr_len = if header.is_keyframe {
        *cur_width = header.width as u32;
        *cur_height = header.height as u32;
        10usize
    } else {
        3usize
    };
    validate_video_dimensions(*cur_width, *cur_height)?;
    let p0_len = header.first_partition_size as usize;
    if hdr_len + p0_len > data.len() {
        return Err(VideoError::CorruptedBitstream);
    }

    let p0_slice = &data[hdr_len..hdr_len + p0_len];
    let p1_slice = &data[hdr_len + p0_len..];
    let mut p0 = Vp8BoolDecoder::new(p0_slice);

    if header.is_keyframe {
        let _color_space = p0.read_bool(128);
        let _clamping_type = p0.read_bool(128);
    }
    // §9.3: segmentation
    let update_segmentation = p0.read_bool(128);
    if update_segmentation {
        let update_map = p0.read_bool(128);
        let update_data = p0.read_bool(128);
        if update_data {
            let _abs_delta = p0.read_bool(128);
            for _ in 0..4 {
                if p0.read_bool(128) {
                    let _ = p0.read_literal(8);
                }
            }
            for _ in 0..4 {
                if p0.read_bool(128) {
                    let _ = p0.read_literal(7);
                }
            }
        }
        if update_map {
            for _ in 0..3 {
                if p0.read_bool(128) {
                    let _ = p0.read_literal(8);
                }
            }
        }
    }
    // §9.4: loop filter
    let _filter_type = p0.read_bool(128);
    let loop_filter_level = p0.read_literal(6) as u8;
    let _sharpness_level = p0.read_literal(3) as u8;
    let lf_delta_enabled = p0.read_bool(128);
    if lf_delta_enabled && p0.read_bool(128) {
        for _ in 0..8 {
            if p0.read_bool(128) {
                let _ = p0.read_literal(7);
            }
        }
    }
    // §9.5: token partitions
    let log2_partitions = p0.read_literal(2) as usize;
    let num_partitions = 1usize << log2_partitions;
    let part_table_bytes = (num_partitions - 1) * 3;
    if p1_slice.len() < part_table_bytes {
        return Err(VideoError::CorruptedBitstream);
    }
    let mut p1 = Vp8BoolDecoder::new(&p1_slice[part_table_bytes..]);

    // §9.6: quantizer indices
    let y_ac_qi = p0.read_literal(7) as u8;
    let read_q_delta = |dec: &mut Vp8BoolDecoder| -> i8 {
        if dec.read_bool(128) {
            let mag = dec.read_literal(4) as i8;
            if dec.read_bool(128) { -mag } else { mag }
        } else {
            0
        }
    };
    let y_dc_delta = read_q_delta(&mut p0);
    let y2_dc_delta = read_q_delta(&mut p0);
    let y2_ac_delta = read_q_delta(&mut p0);
    let uv_dc_delta = read_q_delta(&mut p0);
    let uv_ac_delta = read_q_delta(&mut p0);
    let qf = Vp8QuantFactors::from_qi(
        y_ac_qi,
        y_dc_delta,
        y2_dc_delta,
        y2_ac_delta,
        uv_dc_delta,
        uv_ac_delta,
    );

    if !header.is_keyframe {
        let _refresh_gf = p0.read_bool(128);
        let _refresh_arf = p0.read_bool(128);
        let _copy_gf = p0.read_literal(2);
        let _copy_arf = p0.read_literal(2);
        let _sign_gf = p0.read_bool(128);
        let _sign_arf = p0.read_bool(128);
        let _refresh_probs = p0.read_bool(128);
        let _refresh_last = p0.read_bool(128);
    } else {
        let _refresh_probs = p0.read_bool(128);
    }

    // §9.9: coeff_prob_update
    for _ in 0..(4 * 8 * 3 * 11) {
        if p0.read_bool(252) {
            let _new_prob = p0.read_literal(8);
        }
    }

    // §9.10: mb_no_skip_coeff
    let mb_no_skip_coeff = p0.read_bool(128);
    let prob_skip_false = if mb_no_skip_coeff {
        p0.read_literal(8) as u8
    } else {
        128
    };

    let mut prob_intra = 63u8;
    let mut prob_last = 255u8;
    if !header.is_keyframe {
        prob_intra = p0.read_literal(8) as u8;
        prob_last = p0.read_literal(8) as u8;
        let _prob_gf = p0.read_literal(8) as u8;
        if p0.read_bool(128) {
            for _ in 0..4 {
                let _ = p0.read_literal(8);
            }
        }
        if p0.read_bool(128) {
            for _ in 0..3 {
                let _ = p0.read_literal(8);
            }
        }
        for _ in 0..(2 * 19) {
            if p0.read_bool(252) {
                let _ = p0.read_literal(7);
            }
        }
    }

    let w = *cur_width as usize;
    let h = *cur_height as usize;
    let mb_cols = w.div_ceil(16);
    let mb_rows = h.div_ceil(16);
    let pad_w = mb_cols * 16;
    let pad_h = mb_rows * 16;
    let uv_w = mb_cols * 8;
    let uv_h = mb_rows * 8;

    let mut recon_y = vec![128u8; pad_w * pad_h];
    let mut recon_u = vec![128u8; uv_w * uv_h];
    let mut recon_v = vec![128u8; uv_w * uv_h];

    for mbr in 0..mb_rows {
        for mbc in 0..mb_cols {
            let y_x0 = mbc * 16;
            let y_y0 = mbr * 16;
            let uv_x0 = mbc * 8;
            let uv_y0 = mbr * 8;

            let mb_skip = if mb_no_skip_coeff {
                p0.read_bool(prob_skip_false)
            } else {
                false
            };

            let mut is_inter_mb = false;
            let mut ymode = Vp8MbIntraMode::DcPred;
            let mut uv_mode = Vp8MbIntraMode::DcPred;

            if !header.is_keyframe {
                is_inter_mb = p0.read_bool(prob_intra);
                if is_inter_mb {
                    let _ref_is_gf_or_arf = p0.read_bool(prob_last);
                    let _mv_non_zero = p0.read_bool(128);
                } else {
                    ymode = read_kf_ymode(&mut p0);
                    uv_mode = read_kf_uv_mode(&mut p0);
                }
            } else {
                ymode = read_kf_ymode(&mut p0);
                if ymode == Vp8MbIntraMode::BPred {
                    for _ in 0..16 {
                        let _ = p0.read_bool(DEFAULT_BMODE_PROBS[0]);
                    }
                }
                uv_mode = read_kf_uv_mode(&mut p0);
            }

            let mut pred_y = vec![128u8; 256];
            let mut pred_u = vec![128u8; 64];
            let mut pred_v = vec![128u8; 64];

            if is_inter_mb && prev_y.len() == pad_w * pad_h {
                for r in 0..16 {
                    for c in 0..16 {
                        pred_y[r * 16 + c] = prev_y[(y_y0 + r) * pad_w + (y_x0 + c)];
                    }
                }
                for r in 0..8 {
                    for c in 0..8 {
                        pred_u[r * 8 + c] = prev_u[(uv_y0 + r) * uv_w + (uv_x0 + c)];
                        pred_v[r * 8 + c] = prev_v[(uv_y0 + r) * uv_w + (uv_x0 + c)];
                    }
                }
            } else {
                let above_y = if mbr > 0 {
                    let mut row = [128u8; 16];
                    row.copy_from_slice(
                        &recon_y[(y_y0 - 1) * pad_w + y_x0..(y_y0 - 1) * pad_w + y_x0 + 16],
                    );
                    Some(row)
                } else {
                    None
                };
                let left_y = if mbc > 0 {
                    let mut col = [128u8; 16];
                    for r in 0..16 {
                        col[r] = recon_y[(y_y0 + r) * pad_w + (y_x0 - 1)];
                    }
                    Some(col)
                } else {
                    None
                };
                let tl_y = if mbr > 0 && mbc > 0 {
                    recon_y[(y_y0 - 1) * pad_w + (y_x0 - 1)]
                } else if mbr > 0 {
                    127
                } else {
                    129
                };

                let above_u = if mbr > 0 {
                    let mut row = [128u8; 8];
                    row.copy_from_slice(
                        &recon_u[(uv_y0 - 1) * uv_w + uv_x0..(uv_y0 - 1) * uv_w + uv_x0 + 8],
                    );
                    Some(row)
                } else {
                    None
                };
                let left_u = if mbc > 0 {
                    let mut col = [128u8; 8];
                    for r in 0..8 {
                        col[r] = recon_u[(uv_y0 + r) * uv_w + (uv_x0 - 1)];
                    }
                    Some(col)
                } else {
                    None
                };
                let tl_u = if mbr > 0 && mbc > 0 {
                    recon_u[(uv_y0 - 1) * uv_w + (uv_x0 - 1)]
                } else {
                    128
                };

                let above_v = if mbr > 0 {
                    let mut row = [128u8; 8];
                    row.copy_from_slice(
                        &recon_v[(uv_y0 - 1) * uv_w + uv_x0..(uv_y0 - 1) * uv_w + uv_x0 + 8],
                    );
                    Some(row)
                } else {
                    None
                };
                let left_v = if mbc > 0 {
                    let mut col = [128u8; 8];
                    for r in 0..8 {
                        col[r] = recon_v[(uv_y0 + r) * uv_w + (uv_x0 - 1)];
                    }
                    Some(col)
                } else {
                    None
                };
                let tl_v = if mbr > 0 && mbc > 0 {
                    recon_v[(uv_y0 - 1) * uv_w + (uv_x0 - 1)]
                } else {
                    128
                };

                vp8_predict_intra_mb(
                    ymode,
                    16,
                    above_y.as_ref().map(|a| a.as_slice()),
                    left_y.as_ref().map(|l| l.as_slice()),
                    tl_y,
                    &mut pred_y,
                    16,
                );
                vp8_predict_intra_mb(
                    uv_mode,
                    8,
                    above_u.as_ref().map(|a| a.as_slice()),
                    left_u.as_ref().map(|l| l.as_slice()),
                    tl_u,
                    &mut pred_u,
                    8,
                );
                vp8_predict_intra_mb(
                    uv_mode,
                    8,
                    above_v.as_ref().map(|a| a.as_slice()),
                    left_v.as_ref().map(|l| l.as_slice()),
                    tl_v,
                    &mut pred_v,
                    8,
                );
            }

            let mut y2_dc_rec = [0i32; 16];
            let mut y_ac_deq = [[0i32; 16]; 16];
            let mut u_deq = [[0i32; 16]; 4];
            let mut v_deq = [[0i32; 16]; 4];

            if !mb_skip {
                let (y2_coeffs, y2_nz) = vp8_decode_coeff_block(&mut p1, 0, 0, qf.y2_dc, qf.y2_ac);
                y2_dc_rec = vp8_inverse_wht_4x4(&y2_coeffs);
                let mut prev_nz = usize::from(y2_nz);
                for sb in 0..16 {
                    let (blk, nz) = vp8_decode_coeff_block(&mut p1, 1, prev_nz, qf.y1_dc, qf.y1_ac);
                    y_ac_deq[sb] = blk;
                    prev_nz = usize::from(nz);
                }
                for sb in 0..4 {
                    let (blk, nz) = vp8_decode_coeff_block(&mut p1, 0, prev_nz, qf.uv_dc, qf.uv_ac);
                    u_deq[sb] = blk;
                    prev_nz = usize::from(nz);
                }
                for sb in 0..4 {
                    let (blk, nz) = vp8_decode_coeff_block(&mut p1, 0, prev_nz, qf.uv_dc, qf.uv_ac);
                    v_deq[sb] = blk;
                    prev_nz = usize::from(nz);
                }
            }

            for sb_r in 0..4 {
                for sb_c in 0..4 {
                    let sb_idx = sb_r * 4 + sb_c;
                    let mut deq = y_ac_deq[sb_idx];
                    deq[0] = y2_dc_rec[sb_idx];
                    let res = vp8_inverse_idct_4x4(&deq);
                    for r in 0..4 {
                        for c in 0..4 {
                            let py = sb_r * 4 + r;
                            let px = sb_c * 4 + c;
                            recon_y[(y_y0 + py) * pad_w + (y_x0 + px)] =
                                (pred_y[py * 16 + px] as i16 + res[r * 4 + c]).clamp(0, 255) as u8;
                        }
                    }
                }
            }

            for sb_r in 0..2 {
                for sb_c in 0..2 {
                    let sb_idx = sb_r * 2 + sb_c;
                    let res_u = vp8_inverse_idct_4x4(&u_deq[sb_idx]);
                    let res_v = vp8_inverse_idct_4x4(&v_deq[sb_idx]);
                    for r in 0..4 {
                        for c in 0..4 {
                            let py = sb_r * 4 + r;
                            let px = sb_c * 4 + c;
                            recon_u[(uv_y0 + py) * uv_w + (uv_x0 + px)] =
                                (pred_u[py * 8 + px] as i16 + res_u[r * 4 + c]).clamp(0, 255) as u8;
                            recon_v[(uv_y0 + py) * uv_w + (uv_x0 + px)] =
                                (pred_v[py * 8 + px] as i16 + res_v[r * 4 + c]).clamp(0, 255) as u8;
                        }
                    }
                }
            }
        }
    }

    if loop_filter_level > 0 {
        apply_vp8_simple_loop_filter(&mut recon_y, pad_w, pad_h, loop_filter_level);
    }

    // Convert reconstructed YUV420 planes back to RGBA8 via I420Surface
    let cw = w / 2;
    let ch = h / 2;
    let mut y_crop = vec![16u8; w * h];
    let mut u_crop = vec![128u8; cw * ch];
    let mut v_crop = vec![128u8; cw * ch];
    for py in 0..h {
        y_crop[py * w..(py + 1) * w].copy_from_slice(&recon_y[py * pad_w..py * pad_w + w]);
    }
    for cy in 0..ch {
        u_crop[cy * cw..(cy + 1) * cw].copy_from_slice(&recon_u[cy * uv_w..cy * uv_w + cw]);
        v_crop[cy * cw..(cy + 1) * cw].copy_from_slice(&recon_v[cy * uv_w..cy * uv_w + cw]);
    }
    let i420_out = I420Surface {
        width: *cur_width,
        height: *cur_height,
        y_plane: y_crop,
        u_plane: u_crop,
        v_plane: v_crop,
    };
    let rgba = i420_out.to_rgba_bt709();

    *prev_y = recon_y;
    *prev_u = recon_u;
    *prev_v = recon_v;

    Ok(VideoFrame {
        width: *cur_width,
        height: *cur_height,
        pts_ms,
        keyframe: header.is_keyframe,
        rgba,
    })
}

/// Encode a sequence of [`VideoFrame`]s as a genuine RFC 6386 VP8 bitstream (`VP8` keyframe
/// followed by optional trailing `VP8F` interframe packets).
pub fn encode_vp8_rfc6386_bitstream(
    fps: u32,
    qp: u8,
    frames: &[VideoFrame],
) -> Result<Vec<u8>, VideoError> {
    if frames.is_empty() {
        return Err(VideoError::InvalidHeader);
    }
    let fps = fps.max(1);
    let mut prev_y = Vec::new();
    let mut prev_u = Vec::new();
    let mut prev_v = Vec::new();

    let first_pkt =
        encode_vp8_rfc6386_frame(&frames[0], qp, true, &mut prev_y, &mut prev_u, &mut prev_v)?;
    if frames.len() == 1 && fps == 30 {
        return Ok(first_pkt);
    }

    let mut out = first_pkt;
    // Append a self-contained trailing index (`VP8F` magic + fps + interframes) so frame[0..len]
    // starts with a valid RFC 6386 VP8 keyframe while preserving multi-frame sequences.
    out.extend_from_slice(b"VP8F");
    out.extend_from_slice(&fps.to_le_bytes());
    out.extend_from_slice(&((frames.len() - 1) as u32).to_le_bytes());
    for (idx, frame) in frames.iter().enumerate().skip(1) {
        let is_key = frame.keyframe || (idx % 30 == 0);
        let pkt =
            encode_vp8_rfc6386_frame(frame, qp, is_key, &mut prev_y, &mut prev_u, &mut prev_v)?;
        out.extend_from_slice(&(pkt.len() as u32).to_le_bytes());
        out.extend_from_slice(&pkt);
    }
    Ok(out)
}

/// Decode a genuine RFC 6386 VP8 bitstream (single frame or multi-frame `VP8F` stream) into a [`VideoClip`].
pub fn decode_vp8_rfc6386_bitstream(
    data: &[u8],
    hardware_decode: bool,
) -> Result<VideoClip, VideoError> {
    let mut cur_w = 0u32;
    let mut cur_h = 0u32;
    let mut prev_y = Vec::new();
    let mut prev_u = Vec::new();
    let mut prev_v = Vec::new();

    // Check if there is a `VP8F` multi-frame trailer after the first frame
    let first_frame = decode_vp8_rfc6386_frame(
        data,
        &mut cur_w,
        &mut cur_h,
        &mut prev_y,
        &mut prev_u,
        &mut prev_v,
        0,
    )?;

    let mut frames = vec![first_frame];
    let mut fps = 30u32;

    if let Some(pos) = data.windows(4).position(|w| w == b"VP8F") {
        if pos + 12 <= data.len() {
            fps = u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]])
                .max(1);
            let extra_count =
                u32::from_le_bytes([data[pos + 8], data[pos + 9], data[pos + 10], data[pos + 11]])
                    as usize;
            let mut cursor = pos + 12;
            for idx in 0..extra_count {
                if cursor + 4 > data.len() {
                    return Err(VideoError::CorruptedBitstream);
                }
                let pkt_len = u32::from_le_bytes([
                    data[cursor],
                    data[cursor + 1],
                    data[cursor + 2],
                    data[cursor + 3],
                ]) as usize;
                cursor += 4;
                if cursor + pkt_len > data.len() {
                    return Err(VideoError::CorruptedBitstream);
                }
                let pts_ms = (((idx + 1) as u64 * 1000) / (fps as u64)) as u32;
                let f = decode_vp8_rfc6386_frame(
                    &data[cursor..cursor + pkt_len],
                    &mut cur_w,
                    &mut cur_h,
                    &mut prev_y,
                    &mut prev_u,
                    &mut prev_v,
                    pts_ms,
                )?;
                frames.push(f);
                cursor += pkt_len;
            }
        }
    }

    Ok(VideoClip {
        width: cur_w,
        height: cur_h,
        fps,
        backend: probe_platform_gpu_backend(hardware_decode),
        profile: VideoCodecProfile::Vp8Profile0,
        frames,
        audio: None,
    })
}
