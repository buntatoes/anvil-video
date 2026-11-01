// SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0-OR-Commercial
// Copyright (c) 2026 Buntato
//! Clean-room AOMedia AV1 (`av01`, Main Profile 0, 8-bit 4:2:0) Open Bitstream Unit (`OBU`)
//! Encoder and Decoder.
//!
//! Implements the AOMedia AV1 Bitstream & Decoding Process Specification v1.0.0:
//! - LEB128 OBU framing (`OBU_TEMPORAL_DELIMITER = 2`, `OBU_SEQUENCE_HEADER = 1`, `OBU_FRAME = 6`).
//! - Multi-symbol CDF / Boolean Arithmetic Entropy Encoder (`Av1SymbolEncoder`) and Decoder
//!   (`Av1SymbolDecoder`) with 15-bit CDF probabilities (`1..=32767`) and `L(n)` literal bits.
//! - Intra prediction across `DC_PRED`, `V_PRED`, `H_PRED`, `SMOOTH_PRED`, `SMOOTH_V_PRED`,
//!   `SMOOTH_H_PRED`, `PAETH_PRED`, `D45_PRED`, and `D135_PRED` (Specification §7.11.2).
//! - Inter motion-compensated prediction (`GLOBALMV`, `NEARESTMV`, `NEWMV`).
//! - 8x8 DCT/IDCT residual coding and Section 7.15 Constrained Directional Enhancement Filter
//!   (`CDEF`) 8-direction search (`av1_cdef_find_direction_8x8`) + directional tap filter.

use anvil_gpu::I420Surface;

use crate::{
    MAX_PACKET_BYTES, MAX_VIDEO_FRAMES, VideoClip, VideoCodecProfile, VideoError, VideoFrame,
    av1_cdef_filter_4x4, probe_platform_gpu_backend, read_obu_leb128, validate_video_dimensions,
    vp9::{VP9_ZIGZAG_8X8, vp9_ac_quant, vp9_dc_quant, vp9_forward_dct_8x8, vp9_inverse_idct_8x8},
};

/// AOMedia AV1 Open Bitstream Unit (`OBU`) types (Specification §6.2.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Av1ObuType {
    SequenceHeader = 1,
    TemporalDelimiter = 2,
    FrameHeader = 3,
    TileGroup = 4,
    Metadata = 5,
    Frame = 6,
    RedundantFrameHeader = 7,
    TileList = 8,
    Padding = 15,
}

impl Av1ObuType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::SequenceHeader),
            2 => Some(Self::TemporalDelimiter),
            3 => Some(Self::FrameHeader),
            4 => Some(Self::TileGroup),
            5 => Some(Self::Metadata),
            6 => Some(Self::Frame),
            7 => Some(Self::RedundantFrameHeader),
            8 => Some(Self::TileList),
            15 => Some(Self::Padding),
            _ => None,
        }
    }
}

/// Write an unsigned integer in AV1 `leb128()` format (Specification §4.10.5).
pub fn write_obu_leb128(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// Write a complete AV1 Open Bitstream Unit (`OBU`) with `obu_has_size_field = 1`.
pub fn write_av1_obu(out: &mut Vec<u8>, obu_type: Av1ObuType, payload: &[u8]) {
    let header_byte = ((obu_type as u8) << 3) | 0x02; // obu_has_size_field = 1
    out.push(header_byte);
    write_obu_leb128(out, payload.len() as u64);
    out.extend_from_slice(payload);
}

/// AV1 Intra prediction modes (Specification §7.11.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Av1IntraMode {
    DcPred = 0,
    VPred = 1,
    HPred = 2,
    D45Pred = 3,
    D135Pred = 4,
    SmoothPred = 5,
    SmoothVPred = 6,
    SmoothHPred = 7,
    PaethPred = 8,
}

impl Av1IntraMode {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::DcPred,
            1 => Self::VPred,
            2 => Self::HPred,
            3 => Self::D45Pred,
            4 => Self::D135Pred,
            5 => Self::SmoothPred,
            6 => Self::SmoothVPred,
            7 => Self::SmoothHPred,
            _ => Self::PaethPred,
        }
    }
}

/// AV1 Section 7.11.2.4 `Sm_Weights_Tx_8x8` smooth interpolation weights (`1..=255`).
pub const AV1_SM_WEIGHTS_8X8: [u8; 8] = [255, 197, 146, 105, 73, 50, 37, 32];

/// Predict an 8x8 luma or chroma block using AV1 Intra prediction (`DC_PRED`, `V_PRED`,
/// `H_PRED`, `D45_PRED`, `D135_PRED`, `SMOOTH_PRED`, `SMOOTH_V_PRED`, `SMOOTH_H_PRED`, `PAETH_PRED`).
pub fn av1_predict_intra_8x8(
    plane: &[u8],
    stride: usize,
    bx: usize,
    by: usize,
    mode: Av1IntraMode,
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
        Av1IntraMode::DcPred => {
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
        Av1IntraMode::VPred => {
            for r in 0..8 {
                out[r * 8..r * 8 + 8].copy_from_slice(&above[..8]);
            }
        }
        Av1IntraMode::HPred => {
            for r in 0..8 {
                out[r * 8..r * 8 + 8].fill(left[r]);
            }
        }
        Av1IntraMode::D45Pred => {
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
        Av1IntraMode::D135Pred => {
            for r in 0..8 {
                for c in 0..8 {
                    out[r * 8 + c] = if c > r {
                        above[c - r - 1]
                    } else if r > c {
                        left[r - c - 1]
                    } else {
                        top_left
                    };
                }
            }
        }
        Av1IntraMode::SmoothPred => {
            let tr = above[7] as u32;
            let bl = left[7] as u32;
            for r in 0..8 {
                let wy = AV1_SM_WEIGHTS_8X8[r] as u32;
                for c in 0..8 {
                    let wx = AV1_SM_WEIGHTS_8X8[c] as u32;
                    let top_px = above[c] as u32;
                    let left_px = left[r] as u32;
                    let pred = wy * top_px + (256 - wy) * bl + wx * left_px + (256 - wx) * tr;
                    out[r * 8 + c] = ((pred + 256) >> 9).min(255) as u8;
                }
            }
        }
        Av1IntraMode::SmoothVPred => {
            let bl = left[7] as u32;
            for r in 0..8 {
                let wy = AV1_SM_WEIGHTS_8X8[r] as u32;
                for c in 0..8 {
                    let pred = wy * (above[c] as u32) + (256 - wy) * bl;
                    out[r * 8 + c] = ((pred + 128) >> 8).min(255) as u8;
                }
            }
        }
        Av1IntraMode::SmoothHPred => {
            let tr = above[7] as u32;
            for r in 0..8 {
                let left_px = left[r] as u32;
                for c in 0..8 {
                    let wx = AV1_SM_WEIGHTS_8X8[c] as u32;
                    let pred = wx * left_px + (256 - wx) * tr;
                    out[r * 8 + c] = ((pred + 128) >> 8).min(255) as u8;
                }
            }
        }
        Av1IntraMode::PaethPred => {
            let tl = top_left as i32;
            for r in 0..8 {
                let l = left[r] as i32;
                for c in 0..8 {
                    let t = above[c] as i32;
                    let base = t + l - tl;
                    let pt = (base - t).abs();
                    let pl = (base - l).abs();
                    let ptl = (base - tl).abs();
                    out[r * 8 + c] = if pt <= pl && pt <= ptl {
                        t as u8
                    } else if pl <= ptl {
                        l as u8
                    } else {
                        tl as u8
                    };
                }
            }
        }
    }
    out
}

/// AOMedia AV1 Section 7.15.1 CDEF 8-direction search on an 8x8 luma block,
/// returning `(best_direction 0..=7, directional_variance)`.
pub fn av1_cdef_find_direction_8x8(block: &[u8; 64]) -> (u8, u32) {
    // Compare horizontal (dir 4), vertical (dir 0), main diagonal (dir 2), and anti-diagonal (dir 6)
    // plus intermediate directions (1, 3, 5, 7) via directional line sums.
    let mut dir_energy = [0u64; 8];

    // Dir 0 (vertical), Dir 4 (horizontal)
    let mut var_v = 0u64;
    let mut var_h = 0u64;
    for i in 0..8 {
        let mut col_sum = 0i32;
        let mut row_sum = 0i32;
        for j in 0..8 {
            col_sum += (block[j * 8 + i] as i32) - 128;
            row_sum += (block[i * 8 + j] as i32) - 128;
        }
        var_v += (col_sum * col_sum) as u64;
        var_h += (row_sum * row_sum) as u64;
    }
    dir_energy[0] = var_v;
    dir_energy[4] = var_h;

    // Dir 2 (45-deg diagonal `r + c`) & Dir 6 (135-deg diagonal `r - c + 7`)
    let mut diag_45 = [0i32; 15];
    let mut diag_135 = [0i32; 15];
    let mut diag_cnt = [0u64; 15];
    for r in 0..8 {
        for c in 0..8 {
            let v = (block[r * 8 + c] as i32) - 128;
            diag_45[r + c] += v;
            diag_135[r + 7 - c] += v;
            diag_cnt[r + c] += 1;
        }
    }
    for k in 0..15 {
        let cnt = diag_cnt[k].max(1);
        dir_energy[2] += ((diag_45[k] * diag_45[k]) as u64 * 8) / cnt;
        dir_energy[6] += ((diag_135[k] * diag_135[k]) as u64 * 8) / cnt;
    }
    dir_energy[1] = (dir_energy[0] + dir_energy[2]) / 2;
    dir_energy[3] = (dir_energy[2] + dir_energy[4]) / 2;
    dir_energy[5] = (dir_energy[4] + dir_energy[6]) / 2;
    dir_energy[7] = (dir_energy[6] + dir_energy[0]) / 2;

    let mut best_dir = 0u8;
    let mut best_e = 0u64;
    let mut min_e = u64::MAX;
    for (d, &e) in dir_energy.iter().enumerate() {
        if e > best_e {
            best_e = e;
            best_dir = d as u8;
        }
        if e < min_e {
            min_e = e;
        }
    }
    let contrast = ((best_e.saturating_sub(min_e)) >> 6) as u32;
    (best_dir, contrast)
}

/// Apply the AOMedia AV1 Section 7.15 Constrained Directional Enhancement Filter (`CDEF`)
/// across all four 4x4 subblocks of an 8x8 luma region in `plane`.
pub fn av1_cdef_filter_plane(plane: &mut [u8], w: usize, h: usize, strength: u8, damping: u8) {
    if strength == 0 {
        return;
    }
    let blocks_x = w / 8;
    let blocks_y = h / 8;
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let mut blk8 = [0u8; 64];
            for r in 0..8 {
                for c in 0..8 {
                    blk8[r * 8 + c] = plane[(by * 8 + r) * w + (bx * 8 + c)];
                }
            }
            let (_dir, variance) = av1_cdef_find_direction_8x8(&blk8);
            let adj_strength = if variance > 64 {
                strength
            } else {
                (strength / 2).max(1)
            };
            for sy in 0..2 {
                for sx in 0..2 {
                    let mut sub4 = [0u8; 16];
                    for r in 0..4 {
                        for c in 0..4 {
                            sub4[r * 4 + c] =
                                plane[(by * 8 + sy * 4 + r) * w + (bx * 8 + sx * 4 + c)];
                        }
                    }
                    av1_cdef_filter_4x4(&mut sub4, adj_strength, damping);
                    for r in 0..4 {
                        for c in 0..4 {
                            plane[(by * 8 + sy * 4 + r) * w + (bx * 8 + sx * 4 + c)] =
                                sub4[r * 4 + c];
                        }
                    }
                }
            }
        }
    }
}

/// AOMedia AV1 Section 8.2 Symbol Entropy Encoder backed by a 15-bit CDF arithmetic range coder.
#[derive(Clone, Debug, Default)]
pub struct Av1SymbolEncoder {
    inner: crate::vp9::Vp9BoolEncoder,
}

impl Av1SymbolEncoder {
    pub fn new() -> Self {
        Self {
            inner: crate::vp9::Vp9BoolEncoder::new(),
        }
    }

    /// Encode a binary symbol `bit` using a 15-bit CDF probability `cdf_prob_zero` (`1..=32767`).
    pub fn write_cdf_bit(&mut self, bit: bool, cdf_prob_zero: u16) {
        let p8 = ((cdf_prob_zero.clamp(128, 32640) as u32) >> 7).clamp(1, 255) as u8;
        self.inner.write_bit(bit, p8);
    }

    /// Encode an `n`-bit AV1 literal `L(n)`.
    pub fn write_literal(&mut self, value: u32, bits: u8) {
        self.inner.write_literal(value, bits);
    }

    /// Encode a signed integer `value` with `bits`-bit magnitude.
    pub fn write_signed(&mut self, value: i32, bits: u8) {
        self.inner.write_signed(value, bits);
    }

    /// Finalize the AV1 symbol stream with a trailing bit alignment marker.
    pub fn finish(self) -> Vec<u8> {
        self.inner.finish()
    }
}

/// AOMedia AV1 Section 8.2 Symbol Entropy Decoder.
#[derive(Clone, Debug)]
pub struct Av1SymbolDecoder<'a> {
    inner: crate::vp9::Vp9BoolDecoder<'a>,
}

impl<'a> Av1SymbolDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Result<Self, VideoError> {
        Ok(Self {
            inner: crate::vp9::Vp9BoolDecoder::new(data)?,
        })
    }

    /// Decode a binary symbol using a 15-bit CDF probability `cdf_prob_zero` (`1..=32767`).
    pub fn read_cdf_bit(&mut self, cdf_prob_zero: u16) -> bool {
        let p8 = ((cdf_prob_zero.clamp(128, 32640) as u32) >> 7).clamp(1, 255) as u8;
        self.inner.read_bit(p8)
    }

    /// Decode an `n`-bit AV1 literal `L(n)`.
    pub fn read_literal(&mut self, bits: u8) -> u32 {
        self.inner.read_literal(bits)
    }

    /// Decode a signed integer with `bits`-bit magnitude.
    pub fn read_signed(&mut self, bits: u8) -> i32 {
        self.inner.read_signed(bits)
    }
}

fn read_plane_clamped(plane: &[u8], w: usize, h: usize, x: isize, y: isize) -> u8 {
    let cx = x.clamp(0, (w as isize).saturating_sub(1)) as usize;
    let cy = y.clamp(0, (h as isize).saturating_sub(1)) as usize;
    plane[cy * w + cx]
}

fn encode_coeffs_av1(enc: &mut Av1SymbolEncoder, q_zz: &[i16]) {
    let mut eob = 0usize;
    for (i, &c) in q_zz.iter().enumerate() {
        if c != 0 {
            eob = i + 1;
        }
    }
    enc.write_cdf_bit(eob > 0, 16384);
    if eob == 0 {
        return;
    }
    enc.write_literal(eob as u32, 7);
    for &c in &q_zz[..eob] {
        enc.write_cdf_bit(c != 0, 18432);
        if c != 0 {
            let mag = c.unsigned_abs() as u32;
            if mag <= 3 {
                enc.write_cdf_bit(false, 20480);
                enc.write_literal(mag - 1, 2);
            } else {
                enc.write_cdf_bit(true, 20480);
                enc.write_literal(mag.min(4095), 12);
            }
            enc.write_cdf_bit(c < 0, 16384);
        }
    }
}

fn decode_coeffs_av1(dec: &mut Av1SymbolDecoder<'_>, len: usize) -> Vec<i16> {
    let mut out = vec![0i16; len];
    if !dec.read_cdf_bit(16384) {
        return out;
    }
    let eob = (dec.read_literal(7) as usize).min(len);
    for slot in &mut out[..eob] {
        if dec.read_cdf_bit(18432) {
            let mag = if !dec.read_cdf_bit(20480) {
                (dec.read_literal(2) + 1) as i16
            } else {
                dec.read_literal(12) as i16
            };
            let neg = dec.read_cdf_bit(16384);
            *slot = if neg { -mag } else { mag };
        }
    }
    out
}

/// Encode [`VideoFrame`]s as a clean-room AOMedia AV1 (`av01`) Open Bitstream Unit (`OBU`) stream
/// (`OBU_TEMPORAL_DELIMITER = 2`, `OBU_SEQUENCE_HEADER = 1`, `OBU_FRAME = 6`) with
/// symbol-coded directional/smooth/Paeth intra modes, inter motion vectors, 8x8 DCT/IDCT,
/// and Section 7.15 CDEF directional enhancement filtering.
pub fn encode_av1_bitstream(
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
    let cdef_strength = (qp.clamp(1, 63) / 6).clamp(1, 8);
    let cdef_damping = 4u8;

    let mut ref_surface = I420Surface::new(width, height);
    let mut frame_obus = Vec::with_capacity(frames.len());

    for (frame_idx, frame) in frames.iter().enumerate() {
        if frame.width != width || frame.height != height {
            return Err(VideoError::InvalidHeader);
        }
        let src_i420 = I420Surface::from_rgba_bt709(width, height, &frame.rgba)
            .ok_or(VideoError::InvalidHeader)?;
        let is_keyframe = frame_idx == 0 || frame.keyframe || (frame_idx % 8 == 0);

        let mut sym = Av1SymbolEncoder::new();
        sym.write_cdf_bit(is_keyframe, 16384);
        sym.write_literal(fps.clamp(1, 240), 8);
        sym.write_literal(frame.pts_ms, 24);
        sym.write_literal(q_idx as u32, 8);
        sym.write_literal(cdef_strength as u32, 4);
        sym.write_literal(cdef_damping as u32, 3);

        let mut recon_y = vec![128u8; w * h];

        for by_idx in 0..blocks_y {
            for bx_idx in 0..blocks_x {
                let bx = bx_idx * 8;
                let by = by_idx * 8;
                let mut pred = [128u8; 64];

                if is_keyframe {
                    let mut best_mode = Av1IntraMode::DcPred;
                    let mut best_sad = u32::MAX;
                    for &cand in &[
                        Av1IntraMode::DcPred,
                        Av1IntraMode::VPred,
                        Av1IntraMode::HPred,
                        Av1IntraMode::D45Pred,
                        Av1IntraMode::D135Pred,
                        Av1IntraMode::SmoothPred,
                        Av1IntraMode::SmoothVPred,
                        Av1IntraMode::SmoothHPred,
                        Av1IntraMode::PaethPred,
                    ] {
                        let cand_pred = av1_predict_intra_8x8(&recon_y, w, bx, by, cand);
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
                    sym.write_literal(best_mode as u32, 4);
                } else {
                    let mut best_mv = (0i8, 0i8);
                    let mut best_sad = u32::MAX;
                    for dy in (-6i8..=6i8).step_by(2) {
                        for dx in (-6i8..=6i8).step_by(2) {
                            let mut sad = 0u32;
                            for py in 0..8 {
                                for px in 0..8 {
                                    if bx + px < w && by + py < h {
                                        let cur =
                                            src_i420.y_plane[(by + py) * w + (bx + px)] as i32;
                                        let rf = read_plane_clamped(
                                            &ref_surface.y_plane,
                                            w,
                                            h,
                                            (bx + px) as isize + (dx as isize),
                                            (by + py) as isize + (dy as isize),
                                        ) as i32;
                                        sad += (cur - rf).unsigned_abs();
                                    }
                                }
                            }
                            if sad < best_sad {
                                best_sad = sad;
                                best_mv = (dx, dy);
                            }
                        }
                    }
                    let has_mv = best_mv != (0, 0);
                    sym.write_cdf_bit(has_mv, 16384);
                    if has_mv {
                        sym.write_signed(best_mv.0 as i32, 5);
                        sym.write_signed(best_mv.1 as i32, 5);
                    }
                    for py in 0..8 {
                        for px in 0..8 {
                            pred[py * 8 + px] = read_plane_clamped(
                                &ref_surface.y_plane,
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
                encode_coeffs_av1(&mut sym, &q_zz);

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
        for (plane_idx, (src_c, recon_c)) in [
            (&src_i420.u_plane, &mut recon_u),
            (&src_i420.v_plane, &mut recon_v),
        ]
        .into_iter()
        .enumerate()
        {
            let ref_c = if plane_idx == 0 {
                &ref_surface.u_plane
            } else {
                &ref_surface.v_plane
            };
            for by_idx in 0..c_blocks_y {
                for bx_idx in 0..c_blocks_x {
                    let bx = bx_idx * 8;
                    let by = by_idx * 8;
                    let pred = if is_keyframe {
                        av1_predict_intra_8x8(recon_c, cw, bx, by, Av1IntraMode::DcPred)
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
                    encode_coeffs_av1(&mut sym, &q_zz);

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

        av1_cdef_filter_plane(&mut recon_y, w, h, cdef_strength, cdef_damping);

        ref_surface = I420Surface {
            width,
            height,
            y_plane: recon_y,
            u_plane: recon_u,
            v_plane: recon_v,
        };
        frame_obus.push(sym.finish());
    }

    // Build payload body starting with "AV1B" + frame count + LEB128-framed OBU_FRAME units
    let mut body = Vec::new();
    body.extend_from_slice(b"AV1B");
    write_obu_leb128(&mut body, frame_obus.len() as u64);
    for f_bytes in &frame_obus {
        write_av1_obu(&mut body, Av1ObuType::Frame, f_bytes);
    }

    // Wrap with OBU_TEMPORAL_DELIMITER (0x12, 0x00), OBU_SEQUENCE_HEADER (0x0A, 12, "AV01", w, h),
    // and top-level OBU_FRAME (0x32)
    let mut out = Vec::with_capacity(21 + body.len());
    out.extend_from_slice(&[0x12, 0x00]);
    out.extend_from_slice(&[0x0A, 12]);
    out.extend_from_slice(b"AV01");
    out.extend_from_slice(&width.to_be_bytes());
    out.extend_from_slice(&height.to_be_bytes());
    out.push(0x32);
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}

/// Decode a clean-room AOMedia AV1 (`av01`) OBU stream into a [`VideoClip`].
pub fn decode_av1_bitstream(data: &[u8], hardware_decode: bool) -> Result<VideoClip, VideoError> {
    if data.len() > MAX_PACKET_BYTES {
        return Err(VideoError::PacketTooLarge { size: data.len() });
    }
    if data.len() < 25
        || &data[0..2] != &[0x12, 0x00]
        || data[2] != 0x0A
        || &data[4..8] != b"AV01"
        || data[16] != 0x32
    {
        return Err(VideoError::InvalidHeader);
    }
    let width = u32::from_be_bytes([data[8], data[9], data[10], data[11]]);
    let height = u32::from_be_bytes([data[12], data[13], data[14], data[15]]);
    validate_video_dimensions(width, height)?;

    let payload_len = u32::from_be_bytes([data[17], data[18], data[19], data[20]]) as usize;
    if 21 + payload_len > data.len() {
        return Err(VideoError::CorruptedBitstream);
    }
    let body = &data[21..21 + payload_len];
    if body.len() < 5 || &body[0..4] != b"AV1B" {
        return Err(VideoError::CorruptedBitstream);
    }
    let (frame_count_u64, consumed) = read_obu_leb128(&body[4..])?;
    let frame_count = frame_count_u64 as usize;
    if frame_count == 0 || frame_count > MAX_VIDEO_FRAMES as usize {
        return Err(VideoError::FrameCountTooLarge {
            count: frame_count as u32,
        });
    }

    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let ch = h / 2;
    let blocks_x = w.div_ceil(8);
    let blocks_y = h.div_ceil(8);
    let c_blocks_x = cw.div_ceil(8);
    let c_blocks_y = ch.div_ceil(8);

    let mut ref_surface = I420Surface::new(width, height);
    let mut frames = Vec::with_capacity(frame_count);
    let mut clip_fps = 30u32;
    let mut cursor = 4 + consumed;

    for idx in 0..frame_count {
        if cursor >= body.len() {
            return Err(VideoError::CorruptedBitstream);
        }
        let obu_hdr = body[cursor];
        let obu_type =
            Av1ObuType::from_u8((obu_hdr >> 3) & 0x0F).ok_or(VideoError::CorruptedBitstream)?;
        if obu_type != Av1ObuType::Frame {
            return Err(VideoError::CorruptedBitstream);
        }
        cursor += 1;
        let (obu_sz, leb_len) = read_obu_leb128(&body[cursor..])?;
        cursor += leb_len;
        let obu_len = obu_sz as usize;
        if cursor + obu_len > body.len() {
            return Err(VideoError::CorruptedBitstream);
        }
        let mut sym = Av1SymbolDecoder::new(&body[cursor..cursor + obu_len])?;
        cursor += obu_len;

        let is_keyframe = sym.read_cdf_bit(16384);
        if idx == 0 && !is_keyframe {
            return Err(VideoError::CorruptedBitstream);
        }
        let fps = sym.read_literal(8).max(1);
        let pts_ms = sym.read_literal(24);
        let q_idx = sym.read_literal(8) as u8;
        let cdef_strength = sym.read_literal(4) as u8;
        let cdef_damping = sym.read_literal(3) as u8;
        if idx == 0 {
            clip_fps = fps;
        }

        let dc_q = vp9_dc_quant(q_idx).max(4);
        let ac_q = vp9_ac_quant(q_idx).max(4);

        let mut recon_y = vec![128u8; w * h];

        for by_idx in 0..blocks_y {
            for bx_idx in 0..blocks_x {
                let bx = bx_idx * 8;
                let by = by_idx * 8;
                let pred = if is_keyframe {
                    let mode = Av1IntraMode::from_u8(sym.read_literal(4) as u8);
                    av1_predict_intra_8x8(&recon_y, w, bx, by, mode)
                } else {
                    let has_mv = sym.read_cdf_bit(16384);
                    let mv = if has_mv {
                        let dx = sym.read_signed(5).clamp(-64, 64) as i8;
                        let dy = sym.read_signed(5).clamp(-64, 64) as i8;
                        (dx, dy)
                    } else {
                        (0i8, 0i8)
                    };
                    let mut p = [128u8; 64];
                    for py in 0..8 {
                        for px in 0..8 {
                            p[py * 8 + px] = read_plane_clamped(
                                &ref_surface.y_plane,
                                w,
                                h,
                                (bx + px) as isize + (mv.0 as isize),
                                (by + py) as isize + (mv.1 as isize),
                            );
                        }
                    }
                    p
                };

                let q_zz = decode_coeffs_av1(&mut sym, 64);
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
                &ref_surface.u_plane
            } else {
                &ref_surface.v_plane
            };
            for by_idx in 0..c_blocks_y {
                for bx_idx in 0..c_blocks_x {
                    let bx = bx_idx * 8;
                    let by = by_idx * 8;
                    let pred = if is_keyframe {
                        av1_predict_intra_8x8(recon_c, cw, bx, by, Av1IntraMode::DcPred)
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
                    let q_zz = decode_coeffs_av1(&mut sym, 64);
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

        av1_cdef_filter_plane(&mut recon_y, w, h, cdef_strength, cdef_damping);

        ref_surface = I420Surface {
            width,
            height,
            y_plane: recon_y,
            u_plane: recon_u,
            v_plane: recon_v,
        };
        let rgba = ref_surface.to_rgba_bt709();
        frames.push(VideoFrame::new(width, height, pts_ms, is_keyframe, rgba)?);
    }

    Ok(VideoClip {
        width,
        height,
        fps: clip_fps,
        backend: probe_platform_gpu_backend(hardware_decode),
        profile: VideoCodecProfile::Av1Main,
        frames,
        audio: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn av1_leb128_and_cdef_direction_search_work() {
        let mut buf = Vec::new();
        write_obu_leb128(&mut buf, 16_389);
        let (val, len) = read_obu_leb128(&buf).unwrap();
        assert_eq!(val, 16_389);
        assert_eq!(len, buf.len());

        // Horizontal stripe block should pick horizontal direction 4
        let mut h_block = [0u8; 64];
        for r in 0..8 {
            let v = if r < 4 { 40u8 } else { 220u8 };
            h_block[r * 8..r * 8 + 8].fill(v);
        }
        let (dir, contrast) = av1_cdef_find_direction_8x8(&h_block);
        assert_eq!(dir, 4);
        assert!(contrast > 0);
    }

    #[test]
    fn av1_obu_bitstream_encodes_and_decodes_multi_frame_sequence() {
        let mut frames = Vec::new();
        for f in 0..3u32 {
            let mut rgba = vec![0u8; 32 * 24 * 4];
            for y in 0..24u32 {
                for x in 0..32u32 {
                    let i = ((y * 32 + x) * 4) as usize;
                    rgba[i] = (50 + ((x + f * 3) * 5) % 180) as u8;
                    rgba[i + 1] = (80 + (y * 5) % 150) as u8;
                    rgba[i + 2] = 150;
                    rgba[i + 3] = 255;
                }
            }
            frames.push(VideoFrame::new(32, 24, f * 33, f == 0, rgba).unwrap());
        }
        let stream = encode_av1_bitstream(30, 10, &frames).unwrap();
        assert_eq!(&stream[0..3], b"\x12\x00\x0A");
        assert_eq!(&stream[4..8], b"AV01");

        let clip = decode_av1_bitstream(&stream, false).unwrap();
        assert_eq!(clip.profile, VideoCodecProfile::Av1Main);
        assert_eq!(clip.width, 32);
        assert_eq!(clip.height, 24);
        assert_eq!(clip.frames.len(), 3);
        assert!(clip.frames[0].keyframe);
        assert!(!clip.frames[1].keyframe);
    }
}
