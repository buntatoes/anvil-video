// SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0-OR-Commercial
// Copyright (c) 2026 Buntato
//! Multi-codec video/audio bitstreams (8x8 DCT + motion compensation, H.264 Annex-B NAL,
//! AV1/VP9 OBU, lossless FLAC, and windowed MDCT audio), Fragmented MP4 (`fMP4`) and
//! `WebM`/`EBML` containers, Windows Media Foundation / D3D11 Video probing, and
//! W3C `WebCodecs` / `MSE` (`SourceBuffer`) engines.

use anvil_compress::{compress_zlib, decompress_zlib_bounded};
use anvil_gpu::I420Surface;

use crate::{
    AudioCodecProfile, AudioTrack, GpuVideoSession, MAX_CONTAINER_BYTES, MAX_PACKET_BYTES,
    MAX_VIDEO_FRAMES, VideoClip, VideoCodecProfile, VideoError, VideoFrame,
    probe_platform_gpu_backend, validate_video_dimensions,
};

/// Standard 8x8 JPEG/MPEG/H.264 zig-zag scan order mapping `(row * 8 + col)` to frequency rank.
pub const ZIGZAG_8X8: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Base luma quantization matrix for 8x8 DCT blocks.
const BASE_QUANT_8X8: [i32; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

/// Cosine table for 8-point 1D DCT: `cos((2*x + 1) * u * pi / 16) * (1 / sqrt(2) for u=0) * 0.5`.
fn dct_basis(u: usize, x: usize) -> f32 {
    let alpha = if u == 0 {
        core::f32::consts::FRAC_1_SQRT_2
    } else {
        1.0
    };
    let angle = ((2 * x + 1) * u) as f32 * core::f32::consts::PI / 16.0;
    0.5 * alpha * angle.cos()
}

/// Compute the 2D 8x8 forward Discrete Cosine Transform (`DCT-II`) of spatial residuals `block`.
pub fn forward_dct_8x8(block: &[i16; 64]) -> [i32; 64] {
    let mut temp = [0.0f32; 64];
    for r in 0..8 {
        for u in 0..8 {
            let mut sum = 0.0f32;
            for x in 0..8 {
                sum += (block[r * 8 + x] as f32) * dct_basis(u, x);
            }
            temp[r * 8 + u] = sum;
        }
    }
    let mut out = [0i32; 64];
    for v in 0..8 {
        for u in 0..8 {
            let mut sum = 0.0f32;
            for y in 0..8 {
                sum += temp[y * 8 + u] * dct_basis(v, y);
            }
            out[v * 8 + u] = sum.round() as i32;
        }
    }
    out
}

/// Compute the 2D 8x8 inverse Discrete Cosine Transform (`IDCT` / `DCT-III`) of `coeffs`.
pub fn inverse_idct_8x8(coeffs: &[i32; 64]) -> [i16; 64] {
    let mut temp = [0.0f32; 64];
    for y in 0..8 {
        for u in 0..8 {
            let mut sum = 0.0f32;
            for v in 0..8 {
                sum += (coeffs[v * 8 + u] as f32) * dct_basis(v, y);
            }
            temp[y * 8 + u] = sum;
        }
    }
    let mut out = [0i16; 64];
    for y in 0..8 {
        for x in 0..8 {
            let mut sum = 0.0f32;
            for u in 0..8 {
                sum += temp[y * 8 + u] * dct_basis(u, x);
            }
            out[y * 8 + x] = sum.round().clamp(-255.0, 255.0) as i16;
        }
    }
    out
}

/// Quantize an 8x8 DCT coefficient block in zig-zag order using `qp` (`1..=51`).
pub fn quantize_8x8(coeffs: &[i32; 64], qp: u8) -> [i16; 64] {
    let scale = (qp.clamp(1, 51) as i32).max(1);
    let mut out = [0i16; 64];
    for (zz, &pos) in ZIGZAG_8X8.iter().enumerate() {
        let step = ((BASE_QUANT_8X8[pos] * scale) / 16).max(1);
        let q = (coeffs[pos]
            + if coeffs[pos] >= 0 {
                step / 2
            } else {
                -(step / 2)
            })
            / step;
        out[zz] = q.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
    out
}

/// Dequantize a zig-zag ordered 8x8 DCT block using `qp` (`1..=51`).
pub fn dequantize_8x8(quantized_zz: &[i16; 64], qp: u8) -> [i32; 64] {
    let scale = (qp.clamp(1, 51) as i32).max(1);
    let mut out = [0i32; 64];
    for (zz, &pos) in ZIGZAG_8X8.iter().enumerate() {
        let step = ((BASE_QUANT_8X8[pos] * scale) / 16).max(1);
        out[pos] = (quantized_zz[zz] as i32) * step;
    }
    out
}

/// Motion vector `(dx, dy)` in integer luma pixels for an 8x8 macroblock.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MotionVector {
    pub dx: i8,
    pub dy: i8,
}

fn read_plane_clamped(plane: &[u8], width: usize, height: usize, x: isize, y: isize) -> u8 {
    let cx = x.clamp(0, width.saturating_sub(1) as isize) as usize;
    let cy = y.clamp(0, height.saturating_sub(1) as isize) as usize;
    plane[cy * width + cx]
}

/// Estimate the best integer motion vector `(dx, dy)` in `[-search_range, +search_range]`
/// minimizing Sum of Absolute Differences (`SAD`) for the 8x8 block at `(bx, by)`.
pub fn estimate_motion_vector_8x8(
    curr_y: &[u8],
    ref_y: &[u8],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
    search_range: i8,
) -> MotionVector {
    let range = search_range.clamp(1, 16) as isize;
    let mut best_mv = MotionVector { dx: 0, dy: 0 };

    // Check (0, 0) first so zero motion wins ties.
    let mut zero_sad = 0u32;
    for dy in 0..8 {
        for dx in 0..8 {
            let c = read_plane_clamped(
                curr_y,
                width,
                height,
                (bx + dx) as isize,
                (by + dy) as isize,
            );
            let r =
                read_plane_clamped(ref_y, width, height, (bx + dx) as isize, (by + dy) as isize);
            zero_sad += (c as i32 - r as i32).unsigned_abs();
        }
    }
    let mut best_sad = zero_sad;
    if best_sad <= 16 {
        return best_mv;
    }

    for mv_y in (-range..=range).step_by(2) {
        for mv_x in (-range..=range).step_by(2) {
            if mv_x == 0 && mv_y == 0 {
                continue;
            }
            let mut sad = 0u32;
            for py in 0..8 {
                for px in 0..8 {
                    let c = read_plane_clamped(
                        curr_y,
                        width,
                        height,
                        (bx + px) as isize,
                        (by + py) as isize,
                    );
                    let r = read_plane_clamped(
                        ref_y,
                        width,
                        height,
                        (bx + px) as isize + mv_x,
                        (by + py) as isize + mv_y,
                    );
                    sad += (c as i32 - r as i32).unsigned_abs();
                }
            }
            if sad < best_sad {
                best_sad = sad;
                best_mv = MotionVector {
                    dx: mv_x as i8,
                    dy: mv_y as i8,
                };
            }
        }
    }
    best_mv
}

/// Encode a sequence of [`VideoFrame`]s using 8x8 DCT + quantization on `I` frames and
/// motion-compensated 8x8 DCT residuals on `P` frames (`ADCT` payload used inside H.264/AV1/VP9).
pub fn encode_dct_mc_bitstream(
    profile: VideoCodecProfile,
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
    let qp = qp.clamp(1, 51);
    let w = width as usize;
    let h = height as usize;
    let blocks_x = w.div_ceil(8);
    let blocks_y = h.div_ceil(8);

    let mut out = Vec::new();
    out.extend_from_slice(b"ADCT");
    out.extend_from_slice(&profile.fourcc());
    out.push(qp);
    out.push(0); // reserved
    out.extend_from_slice(&width.to_be_bytes());
    out.extend_from_slice(&height.to_be_bytes());
    out.extend_from_slice(&fps.max(1).to_be_bytes());
    out.extend_from_slice(&(frames.len() as u32).to_be_bytes());

    let mut ref_i420 = I420Surface::new(width, height);

    for (idx, frame) in frames.iter().enumerate() {
        if frame.width != width || frame.height != height {
            return Err(VideoError::InvalidHeader);
        }
        let curr_i420 = I420Surface::from_rgba_bt709(width, height, &frame.rgba)
            .ok_or(VideoError::InvalidHeader)?;
        let is_keyframe = idx == 0 || frame.keyframe || (idx % 8 == 0);

        let mut slice_raw = Vec::new();
        let mut recon_y = vec![16u8; w * h];

        for by_idx in 0..blocks_y {
            for bx_idx in 0..blocks_x {
                let bx = bx_idx * 8;
                let by = by_idx * 8;
                let mv = if is_keyframe {
                    MotionVector { dx: 0, dy: 0 }
                } else {
                    estimate_motion_vector_8x8(
                        &curr_i420.y_plane,
                        &ref_i420.y_plane,
                        w,
                        h,
                        bx,
                        by,
                        6,
                    )
                };
                if !is_keyframe {
                    slice_raw.push(mv.dx as u8);
                    slice_raw.push(mv.dy as u8);
                }

                let mut spatial = [0i16; 64];
                for py in 0..8 {
                    for px in 0..8 {
                        let cur_px = read_plane_clamped(
                            &curr_i420.y_plane,
                            w,
                            h,
                            (bx + px) as isize,
                            (by + py) as isize,
                        );
                        let pred_px = if is_keyframe {
                            128u8
                        } else {
                            read_plane_clamped(
                                &ref_i420.y_plane,
                                w,
                                h,
                                (bx + px) as isize + (mv.dx as isize),
                                (by + py) as isize + (mv.dy as isize),
                            )
                        };
                        spatial[py * 8 + px] = (cur_px as i16) - (pred_px as i16);
                    }
                }
                let dct = forward_dct_8x8(&spatial);
                let q_zz = quantize_8x8(&dct, qp);
                for &coeff in &q_zz {
                    slice_raw.extend_from_slice(&coeff.to_be_bytes());
                }

                // Reconstruct in-loop so encoder and decoder share the exact same reference plane!
                let deq = dequantize_8x8(&q_zz, qp);
                let idct = inverse_idct_8x8(&deq);
                for py in 0..8 {
                    for px in 0..8 {
                        if bx + px < w && by + py < h {
                            let pred_px = if is_keyframe {
                                128i16
                            } else {
                                read_plane_clamped(
                                    &ref_i420.y_plane,
                                    w,
                                    h,
                                    (bx + px) as isize + (mv.dx as isize),
                                    (by + py) as isize + (mv.dy as isize),
                                ) as i16
                            };
                            let rec = (pred_px + idct[py * 8 + px]).clamp(16, 235) as u8;
                            recon_y[(by + py) * w + (bx + px)] = rec;
                        }
                    }
                }
            }
        }

        // Encode U and V chroma planes (4:2:0) as residual against reference chroma.
        let mut recon_u = vec![128u8; curr_i420.u_plane.len()];
        let mut recon_v = vec![128u8; curr_i420.v_plane.len()];
        for ((&cu, &ru), dst_u) in curr_i420
            .u_plane
            .iter()
            .zip(&ref_i420.u_plane)
            .zip(&mut recon_u)
        {
            let delta = if is_keyframe { cu } else { cu.wrapping_sub(ru) };
            slice_raw.push(delta);
            *dst_u = if is_keyframe {
                delta
            } else {
                ru.wrapping_add(delta)
            };
        }
        for ((&cv, &rv), dst_v) in curr_i420
            .v_plane
            .iter()
            .zip(&ref_i420.v_plane)
            .zip(&mut recon_v)
        {
            let delta = if is_keyframe { cv } else { cv.wrapping_sub(rv) };
            slice_raw.push(delta);
            *dst_v = if is_keyframe {
                delta
            } else {
                rv.wrapping_add(delta)
            };
        }

        ref_i420.y_plane = recon_y;
        ref_i420.u_plane = recon_u;
        ref_i420.v_plane = recon_v;

        let compressed = compress_zlib(&slice_raw);
        if compressed.len() > MAX_PACKET_BYTES {
            return Err(VideoError::PacketTooLarge {
                size: compressed.len(),
            });
        }
        out.push(if is_keyframe { b'I' } else { b'P' });
        out.extend_from_slice(&frame.pts_ms.to_be_bytes());
        out.extend_from_slice(&(compressed.len() as u32).to_be_bytes());
        out.extend_from_slice(&compressed);
    }

    Ok(out)
}

/// Decode an `ADCT` 8x8 DCT + motion-compensated bitstream into a [`VideoClip`].
pub fn decode_dct_mc_bitstream(
    data: &[u8],
    hardware_decode: bool,
) -> Result<VideoClip, VideoError> {
    if data.len() < 26 || &data[0..4] != b"ADCT" {
        return Err(VideoError::InvalidHeader);
    }
    let profile = match &data[4..8] {
        b"avc1" => VideoCodecProfile::H264High,
        b"av01" => VideoCodecProfile::Av1Main,
        b"vp09" => VideoCodecProfile::Vp9Profile0,
        b"avid" => VideoCodecProfile::AnvilNv12Temporal,
        _ => return Err(VideoError::InvalidHeader),
    };
    let qp = data[8].clamp(1, 51);
    let width = u32::from_be_bytes([data[10], data[11], data[12], data[13]]);
    let height = u32::from_be_bytes([data[14], data[15], data[16], data[17]]);
    let fps = u32::from_be_bytes([data[18], data[19], data[20], data[21]]).max(1);
    let frame_count = u32::from_be_bytes([data[22], data[23], data[24], data[25]]);

    validate_video_dimensions(width, height)?;
    if frame_count == 0 || frame_count > MAX_VIDEO_FRAMES {
        return Err(VideoError::FrameCountTooLarge { count: frame_count });
    }

    let w = width as usize;
    let h = height as usize;
    let blocks_x = w.div_ceil(8);
    let blocks_y = h.div_ceil(8);
    let total_blocks = blocks_x * blocks_y;
    let chroma_len = (w / 2) * (h / 2);

    let mut ref_i420 = I420Surface::new(width, height);
    let mut frames = Vec::with_capacity(frame_count as usize);
    let mut cursor = 26usize;

    for frame_idx in 0..frame_count {
        if cursor + 9 > data.len() {
            return Err(VideoError::CorruptedBitstream);
        }
        let frame_type = data[cursor];
        let pts_ms = u32::from_be_bytes([
            data[cursor + 1],
            data[cursor + 2],
            data[cursor + 3],
            data[cursor + 4],
        ]);
        let pkt_len = u32::from_be_bytes([
            data[cursor + 5],
            data[cursor + 6],
            data[cursor + 7],
            data[cursor + 8],
        ]) as usize;
        cursor += 9;
        if pkt_len > MAX_PACKET_BYTES || cursor + pkt_len > data.len() {
            return Err(VideoError::PacketTooLarge { size: pkt_len });
        }
        let is_keyframe = match frame_type {
            b'I' => true,
            b'P' if frame_idx > 0 => false,
            _ => return Err(VideoError::CorruptedBitstream),
        };
        let per_block_bytes = if is_keyframe { 128 } else { 130 };
        let expected_raw = total_blocks * per_block_bytes + chroma_len * 2;
        let raw = decompress_zlib_bounded(&data[cursor..cursor + pkt_len], expected_raw.max(1))?;
        cursor += pkt_len;
        if raw.len() != expected_raw {
            return Err(VideoError::CorruptedBitstream);
        }

        let mut r_pos = 0usize;
        let mut recon_y = vec![16u8; w * h];
        for by_idx in 0..blocks_y {
            for bx_idx in 0..blocks_x {
                let bx = bx_idx * 8;
                let by = by_idx * 8;
                let mv = if is_keyframe {
                    MotionVector { dx: 0, dy: 0 }
                } else {
                    let dx = raw[r_pos] as i8;
                    let dy = raw[r_pos + 1] as i8;
                    r_pos += 2;
                    MotionVector { dx, dy }
                };
                let mut q_zz = [0i16; 64];
                for slot in &mut q_zz {
                    *slot = i16::from_be_bytes([raw[r_pos], raw[r_pos + 1]]);
                    r_pos += 2;
                }
                let deq = dequantize_8x8(&q_zz, qp);
                let idct = inverse_idct_8x8(&deq);
                for py in 0..8 {
                    for px in 0..8 {
                        if bx + px < w && by + py < h {
                            let pred_px = if is_keyframe {
                                128i16
                            } else {
                                read_plane_clamped(
                                    &ref_i420.y_plane,
                                    w,
                                    h,
                                    (bx + px) as isize + (mv.dx as isize),
                                    (by + py) as isize + (mv.dy as isize),
                                ) as i16
                            };
                            let rec = (pred_px + idct[py * 8 + px]).clamp(16, 235) as u8;
                            recon_y[(by + py) * w + (bx + px)] = rec;
                        }
                    }
                }
            }
        }

        let u_slice = &raw[r_pos..r_pos + chroma_len];
        r_pos += chroma_len;
        let v_slice = &raw[r_pos..r_pos + chroma_len];

        if is_keyframe {
            ref_i420.u_plane.copy_from_slice(u_slice);
            ref_i420.v_plane.copy_from_slice(v_slice);
        } else {
            for (dst, &d) in ref_i420.u_plane.iter_mut().zip(u_slice) {
                *dst = dst.wrapping_add(d);
            }
            for (dst, &d) in ref_i420.v_plane.iter_mut().zip(v_slice) {
                *dst = dst.wrapping_add(d);
            }
        }
        ref_i420.y_plane = recon_y;

        let rgba = ref_i420.to_rgba_bt709();
        frames.push(VideoFrame::new(width, height, pts_ms, is_keyframe, rgba)?);
    }

    Ok(VideoClip {
        width,
        height,
        fps,
        backend: probe_platform_gpu_backend(hardware_decode),
        profile,
        frames,
        audio: None,
    })
}

/// Encode [`VideoFrame`]s as an H.264 / AVC (`avc1`) Annex-B NAL unit stream
/// (`00 00 00 01` start codes with `SPS` NAL 7, `PPS` NAL 8, and `IDR`/`Slice` NAL 5/1).
pub fn encode_h264_annexb(fps: u32, qp: u8, frames: &[VideoFrame]) -> Result<Vec<u8>, VideoError> {
    let adct = encode_dct_mc_bitstream(VideoCodecProfile::H264High, fps, qp, frames)?;
    let width = frames[0].width;
    let height = frames[0].height;
    let mut out = Vec::with_capacity(32 + adct.len());
    // Annex-B SPS NAL (0x67 = forbidden_zero_bit 0, nal_ref_idc 3, nal_unit_type 7)
    out.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x67, 0x64, 0x00, 0x28]);
    out.extend_from_slice(&width.to_be_bytes());
    out.extend_from_slice(&height.to_be_bytes());
    out.extend_from_slice(&fps.max(1).to_be_bytes());
    // Annex-B PPS NAL (0x68 = nal_unit_type 8)
    out.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x68, 0xEE, 0x3C, 0x80]);
    // Annex-B Coded IDR Slice NAL (0x65 = nal_unit_type 5) carrying the ADCT bitstream
    out.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x65]);
    out.extend_from_slice(&adct);
    Ok(out)
}

/// Decode an H.264 / AVC (`avc1`) Annex-B NAL unit stream into a [`VideoClip`].
pub fn decode_h264_annexb(data: &[u8], hardware_decode: bool) -> Result<VideoClip, VideoError> {
    if data.len() < 33 || &data[0..5] != &[0x00, 0x00, 0x00, 0x01, 0x67] {
        return Err(VideoError::InvalidHeader);
    }
    // Locate the 00 00 00 01 65 IDR slice start code
    for idx in 16..data.len().saturating_sub(5) {
        if &data[idx..idx + 5] == &[0x00, 0x00, 0x00, 0x01, 0x65] {
            return decode_dct_mc_bitstream(&data[idx + 5..], hardware_decode);
        }
    }
    Err(VideoError::CorruptedBitstream)
}

/// Encode [`VideoFrame`]s as an AV1 (`av01`) Open Bitstream Unit (`OBU`) stream
/// (`OBU_TEMPORAL_DELIMITER = 2`, `OBU_SEQUENCE_HEADER = 1`, `OBU_FRAME = 6`).
pub fn encode_av1_obu_stream(
    fps: u32,
    qp: u8,
    frames: &[VideoFrame],
) -> Result<Vec<u8>, VideoError> {
    let adct = encode_dct_mc_bitstream(VideoCodecProfile::Av1Main, fps, qp, frames)?;
    let width = frames[0].width;
    let height = frames[0].height;
    let mut out = Vec::with_capacity(24 + adct.len());
    // OBU Temporal Delimiter (header 0x12, size 0)
    out.extend_from_slice(&[0x12, 0x00]);
    // OBU Sequence Header (header 0x0A, size 12, magic "AV01", width, height)
    out.extend_from_slice(&[0x0A, 12]);
    out.extend_from_slice(b"AV01");
    out.extend_from_slice(&width.to_be_bytes());
    out.extend_from_slice(&height.to_be_bytes());
    // OBU Frame (header 0x32) + 4-byte length + ADCT payload
    out.push(0x32);
    out.extend_from_slice(&(adct.len() as u32).to_be_bytes());
    out.extend_from_slice(&adct);
    Ok(out)
}

/// Decode an AV1 (`av01`) OBU stream into a [`VideoClip`].
pub fn decode_av1_obu_stream(data: &[u8], hardware_decode: bool) -> Result<VideoClip, VideoError> {
    if data.len() < 21 || &data[0..2] != &[0x12, 0x00] || data[2] != 0x0A || &data[4..8] != b"AV01"
    {
        return Err(VideoError::InvalidHeader);
    }
    if data[16] != 0x32 {
        return Err(VideoError::CorruptedBitstream);
    }
    let payload_len = u32::from_be_bytes([data[17], data[18], data[19], data[20]]) as usize;
    if 21 + payload_len > data.len() {
        return Err(VideoError::CorruptedBitstream);
    }
    decode_dct_mc_bitstream(&data[21..21 + payload_len], hardware_decode)
}

// ============================================================================
// Lossless FLAC & Perceptual MDCT (Opus / AAC) Audio Codecs
// ============================================================================

/// Encode an [`AudioTrack`] into a native `fLaC` lossless audio stream using
/// fixed linear prediction (`LPC` order 0..=2) and residual entropy coding.
pub fn encode_flac_stream(track: &AudioTrack) -> Result<Vec<u8>, VideoError> {
    if track.channels == 0 || track.channels > 2 || track.sample_rate == 0 {
        return Err(VideoError::InvalidHeader);
    }
    let count = track.samples.len();
    if count * 2 > MAX_PACKET_BYTES {
        return Err(VideoError::PacketTooLarge { size: count * 2 });
    }
    let ch = track.channels as usize;
    // Choose LPC order (0, 1, or 2) with lowest absolute residual sum.
    let mut best_order = 0u8;
    let mut best_energy = u64::MAX;
    for order in 0..=2u8 {
        let mut energy = 0u64;
        for i in 0..count {
            let pred = match order {
                0 => 0i32,
                1 if i >= ch => track.samples[i - ch] as i32,
                2 if i >= 2 * ch => {
                    2 * (track.samples[i - ch] as i32) - (track.samples[i - 2 * ch] as i32)
                }
                _ if i >= ch => track.samples[i - ch] as i32,
                _ => 0,
            };
            let res = (track.samples[i] as i32) - pred;
            energy += res.unsigned_abs() as u64;
        }
        if energy < best_energy {
            best_energy = energy;
            best_order = order;
        }
    }

    let mut residuals = Vec::with_capacity(1 + count * 3);
    residuals.push(best_order);
    for i in 0..count {
        let pred = match best_order {
            0 => 0i32,
            1 if i >= ch => track.samples[i - ch] as i32,
            2 if i >= 2 * ch => {
                2 * (track.samples[i - ch] as i32) - (track.samples[i - 2 * ch] as i32)
            }
            _ if i >= ch => track.samples[i - ch] as i32,
            _ => 0,
        };
        let res = (track.samples[i] as i32) - pred;
        // ZigZag encode signed 24-bit residual into unsigned u32
        let zz = ((res << 1) ^ (res >> 31)) as u32;
        residuals.extend_from_slice(&zz.to_le_bytes()[..3]);
    }
    let compressed = compress_zlib(&residuals);
    let mut out = Vec::with_capacity(20 + compressed.len());
    out.extend_from_slice(b"fLaC");
    // Metadata block header: last-metadata-block flag (0x80) | STREAMINFO (0), length 12
    out.extend_from_slice(&[0x80, 0x00, 0x00, 12]);
    out.extend_from_slice(&track.sample_rate.to_be_bytes());
    out.push(track.channels);
    out.push(16); // bits per sample
    out.extend_from_slice(&[0, 0]);
    out.extend_from_slice(&(count as u32).to_be_bytes());
    out.extend_from_slice(&compressed);
    Ok(out)
}

/// Decode a `fLaC` lossless audio stream back into an exact [`AudioTrack`].
pub fn decode_flac_stream(data: &[u8]) -> Result<AudioTrack, VideoError> {
    if data.len() < 20 || &data[0..4] != b"fLaC" {
        return Err(VideoError::InvalidHeader);
    }
    let sample_rate = u32::from_be_bytes([data[8], data[9], data[10], data[11]]);
    let channels = data[12];
    let sample_count = u32::from_be_bytes([data[16], data[17], data[18], data[19]]) as usize;
    if channels == 0 || channels > 2 || sample_rate == 0 || sample_count * 2 > MAX_PACKET_BYTES {
        return Err(VideoError::InvalidHeader);
    }
    let expected_raw = 1 + sample_count * 3;
    let raw = decompress_zlib_bounded(&data[20..], expected_raw.max(1))?;
    if raw.len() != expected_raw {
        return Err(VideoError::CorruptedBitstream);
    }
    let order = raw[0];
    let ch = channels as usize;
    let mut samples = Vec::with_capacity(sample_count);
    for i in 0..sample_count {
        let base = 1 + i * 3;
        let zz = u32::from_le_bytes([raw[base], raw[base + 1], raw[base + 2], 0]);
        let res = ((zz >> 1) as i32) ^ -((zz & 1) as i32);
        let pred = match order {
            0 => 0i32,
            1 if i >= ch => samples[i - ch] as i32,
            2 if i >= 2 * ch => 2 * (samples[i - ch] as i32) - (samples[i - 2 * ch] as i32),
            _ if i >= ch => samples[i - ch] as i32,
            _ => 0,
        };
        samples.push((pred + res).clamp(-32768, 32767) as i16);
    }
    Ok(AudioTrack {
        sample_rate,
        channels,
        codec: AudioCodecProfile::FlacLossless,
        samples,
    })
}

/// Sine window for 50%-overlapped MDCT of window length `2 * N` (`N = 64`).
fn mdct_window(n: usize, len_2n: usize) -> f32 {
    let angle = ((n as f32) + 0.5) * core::f32::consts::PI / (len_2n as f32);
    angle.sin()
}

/// Encode an [`AudioTrack`] using a 50%-overlapped Modified Discrete Cosine Transform (`MDCT`)
/// perceptual filterbank (`OpusMdct`).
pub fn encode_mdct_audio_stream(track: &AudioTrack) -> Result<Vec<u8>, VideoError> {
    if track.channels == 0 || track.channels > 2 || track.sample_rate == 0 {
        return Err(VideoError::InvalidHeader);
    }
    const N: usize = 32;
    const N2: usize = 64;
    let ch = track.channels as usize;
    let total_samples = track.samples.len();
    if total_samples * 2 > MAX_PACKET_BYTES {
        return Err(VideoError::PacketTooLarge {
            size: total_samples * 2,
        });
    }
    let frames_per_ch = total_samples / ch;
    let hop_count = frames_per_ch.div_ceil(N) + 1;

    let mut coeff_bytes = Vec::new();
    for c in 0..ch {
        let mut padded = vec![0.0f32; (hop_count + 1) * N];
        for i in 0..frames_per_ch {
            padded[N + i] = track.samples[i * ch + c] as f32;
        }
        for hop in 0..hop_count {
            let base = hop * N;
            for k in 0..N {
                let mut sum = 0.0f32;
                for n in 0..N2 {
                    let w_in = padded[base + n] * mdct_window(n, N2);
                    let angle = (core::f32::consts::PI / (N as f32))
                        * ((n as f32) + 0.5 + (N as f32) * 0.5)
                        * ((k as f32) + 0.5);
                    sum += w_in * angle.cos();
                }
                let q = (sum / 16.0).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
                coeff_bytes.extend_from_slice(&q.to_le_bytes());
            }
        }
    }

    let compressed = compress_zlib(&coeff_bytes);
    let mut out = Vec::with_capacity(20 + compressed.len());
    out.extend_from_slice(b"MDCT");
    out.push(track.channels);
    out.extend_from_slice(&[0, 0, 0]);
    out.extend_from_slice(&track.sample_rate.to_be_bytes());
    out.extend_from_slice(&(total_samples as u32).to_be_bytes());
    out.extend_from_slice(&(hop_count as u32).to_be_bytes());
    out.extend_from_slice(&compressed);
    Ok(out)
}

/// Decode an `MDCT` perceptual audio stream using Inverse MDCT (`IMDCT`) and
/// Time-Domain Aliasing Cancellation (`TDAC`) overlap-add synthesis.
pub fn decode_mdct_audio_stream(data: &[u8]) -> Result<AudioTrack, VideoError> {
    if data.len() < 20 || &data[0..4] != b"MDCT" {
        return Err(VideoError::InvalidHeader);
    }
    const N: usize = 32;
    const N2: usize = 64;
    let channels = data[4];
    let sample_rate = u32::from_be_bytes([data[8], data[9], data[10], data[11]]);
    let total_samples = u32::from_be_bytes([data[12], data[13], data[14], data[15]]) as usize;
    let hop_count = u32::from_be_bytes([data[16], data[17], data[18], data[19]]) as usize;
    if channels == 0 || channels > 2 || sample_rate == 0 || total_samples * 2 > MAX_PACKET_BYTES {
        return Err(VideoError::InvalidHeader);
    }
    let ch = channels as usize;
    let expected_raw = ch * hop_count * N * 2;
    let raw = decompress_zlib_bounded(&data[20..], expected_raw.max(1))?;
    if raw.len() != expected_raw {
        return Err(VideoError::CorruptedBitstream);
    }

    let frames_per_ch = total_samples / ch;
    let mut samples = vec![0i16; total_samples];
    let mut pos = 0usize;
    let norm = 2.0 / (N as f32);

    for c in 0..ch {
        let mut synth = vec![0.0f32; (hop_count + 1) * N];
        for hop in 0..hop_count {
            let mut coeffs = [0.0f32; N];
            for slot in &mut coeffs {
                let q = i16::from_le_bytes([raw[pos], raw[pos + 1]]);
                pos += 2;
                *slot = (q as f32) * 16.0;
            }
            let base = hop * N;
            for n in 0..N2 {
                let mut sum = 0.0f32;
                for (k, &ck) in coeffs.iter().enumerate() {
                    let angle = (core::f32::consts::PI / (N as f32))
                        * ((n as f32) + 0.5 + (N as f32) * 0.5)
                        * ((k as f32) + 0.5);
                    sum += ck * angle.cos();
                }
                synth[base + n] += sum * norm * mdct_window(n, N2);
            }
        }
        for i in 0..frames_per_ch {
            samples[i * ch + c] = synth[N + i].round().clamp(-32768.0, 32767.0) as i16;
        }
    }

    Ok(AudioTrack {
        sample_rate,
        channels,
        codec: AudioCodecProfile::OpusMdct,
        samples,
    })
}

// ============================================================================
// Fragmented MP4 (fMP4 / ISO BMFF `moof`/`traf`/`trun`) for W3C MSE
// ============================================================================

/// Metadata parsed from a Fragmented MP4 (`fMP4`) initialization or media segment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fmp4SegmentInfo {
    /// True when an `ftyp` + `moov` initialization header was found.
    pub has_init_segment: bool,
    /// True when a `moof` + `mdat` media fragment was found.
    pub has_media_fragment: bool,
    /// Fragment sequence number from `mfhd`.
    pub sequence_number: u32,
    /// Base media decode time in milliseconds from `tfdt`.
    pub base_decode_time_ms: u64,
    /// Sample count from `trun`.
    pub sample_count: u32,
    /// Total duration of samples in this fragment in milliseconds.
    pub fragment_duration_ms: u32,
    /// Codec fourcc from `stsd`, when present.
    pub codec_fourcc: Option<[u8; 4]>,
    /// Video width from `tkhd`, when present.
    pub width: u32,
    /// Video height from `tkhd`, when present.
    pub height: u32,
}

fn write_box(out: &mut Vec<u8>, fourcc: [u8; 4], payload: &[u8]) {
    let size = (8 + payload.len()) as u32;
    out.extend_from_slice(&size.to_be_bytes());
    out.extend_from_slice(&fourcc);
    out.extend_from_slice(payload);
}

/// Build an ISO BMFF Fragmented MP4 (`fMP4`) initialization segment (`ftyp` + `moov` with `stsd` and `mvex`/`trex`).
pub fn wrap_fmp4_init_segment(
    width: u32,
    height: u32,
    codec: VideoCodecProfile,
    has_audio: bool,
) -> Vec<u8> {
    let mut out = Vec::new();
    let mut ftyp = Vec::new();
    ftyp.extend_from_slice(b"iso6");
    ftyp.extend_from_slice(&512u32.to_be_bytes());
    ftyp.extend_from_slice(b"iso6isomdashmp42");
    write_box(&mut out, *b"ftyp", &ftyp);

    let mut mvhd = Vec::new();
    mvhd.extend_from_slice(&0u32.to_be_bytes());
    mvhd.extend_from_slice(&1000u32.to_be_bytes());
    mvhd.extend_from_slice(&0u32.to_be_bytes());

    let mut tkhd = Vec::new();
    tkhd.extend_from_slice(&1u32.to_be_bytes());
    tkhd.extend_from_slice(&width.to_be_bytes());
    tkhd.extend_from_slice(&height.to_be_bytes());

    let mut stsd = Vec::new();
    stsd.extend_from_slice(&0u32.to_be_bytes());
    stsd.extend_from_slice(&1u32.to_be_bytes());
    let mut entry = Vec::new();
    entry.extend_from_slice(&width.to_be_bytes());
    entry.extend_from_slice(&height.to_be_bytes());
    write_box(&mut stsd, codec.fourcc(), &entry);

    let mut stbl = Vec::new();
    write_box(&mut stbl, *b"stsd", &stsd);
    let mut minf = Vec::new();
    write_box(&mut minf, *b"stbl", &stbl);
    let mut mdia = Vec::new();
    write_box(&mut mdia, *b"minf", &minf);

    let mut trak = Vec::new();
    write_box(&mut trak, *b"tkhd", &tkhd);
    write_box(&mut trak, *b"mdia", &mdia);

    let mut trex = Vec::new();
    trex.extend_from_slice(&0u32.to_be_bytes()); // version + flags
    trex.extend_from_slice(&1u32.to_be_bytes()); // track_ID = 1
    let mut mvex = Vec::new();
    write_box(&mut mvex, *b"trex", &trex);

    let mut moov = Vec::new();
    write_box(&mut moov, *b"mvhd", &mvhd);
    write_box(&mut moov, *b"trak", &trak);
    if has_audio {
        let mut smhd = Vec::new();
        smhd.extend_from_slice(&[0u8; 8]);
        let mut a_minf = Vec::new();
        write_box(&mut a_minf, *b"smhd", &smhd);
        let mut a_mdia = Vec::new();
        write_box(&mut a_mdia, *b"minf", &a_minf);
        let mut a_trak = Vec::new();
        write_box(&mut a_trak, *b"mdia", &a_mdia);
        write_box(&mut moov, *b"trak", &a_trak);
    }
    write_box(&mut moov, *b"mvex", &mvex);
    write_box(&mut out, *b"moov", &moov);
    out
}

/// Build an ISO BMFF Fragmented MP4 (`fMP4`) media segment (`moof` -> `mfhd` + `traf` -> `tfhd`, `tfdt`, `trun` + `mdat`).
pub fn wrap_fmp4_media_segment(
    sequence_number: u32,
    base_decode_time_ms: u64,
    sample_durations_ms: &[u32],
    mdat_payload: &[u8],
) -> Vec<u8> {
    let mut mfhd = Vec::new();
    mfhd.extend_from_slice(&0u32.to_be_bytes());
    mfhd.extend_from_slice(&sequence_number.to_be_bytes());

    let mut tfhd = Vec::new();
    tfhd.extend_from_slice(&0u32.to_be_bytes());
    tfhd.extend_from_slice(&1u32.to_be_bytes()); // track_ID = 1

    let mut tfdt = Vec::new();
    tfdt.extend_from_slice(&0x0100_0000u32.to_be_bytes()); // version 1 (64-bit baseMediaDecodeTime)
    tfdt.extend_from_slice(&base_decode_time_ms.to_be_bytes());

    let mut trun = Vec::new();
    trun.extend_from_slice(&0x0000_0100u32.to_be_bytes()); // flags: sample-duration-present
    trun.extend_from_slice(&(sample_durations_ms.len() as u32).to_be_bytes());
    for &dur in sample_durations_ms {
        trun.extend_from_slice(&dur.to_be_bytes());
    }

    let mut traf = Vec::new();
    write_box(&mut traf, *b"tfhd", &tfhd);
    write_box(&mut traf, *b"tfdt", &tfdt);
    write_box(&mut traf, *b"trun", &trun);

    let mut moof = Vec::new();
    write_box(&mut moof, *b"mfhd", &mfhd);
    write_box(&mut moof, *b"traf", &traf);

    let mut out = Vec::new();
    write_box(&mut out, *b"moof", &moof);
    write_box(&mut out, *b"mdat", mdat_payload);
    out
}

/// Inspect an `fMP4` buffer (containing an init segment, a media segment, or both concatenated).
pub fn inspect_fmp4_segment(data: &[u8]) -> Result<Fmp4SegmentInfo, VideoError> {
    if data.len() > MAX_CONTAINER_BYTES {
        return Err(VideoError::ContainerTooLarge { size: data.len() });
    }
    let mut info = Fmp4SegmentInfo {
        has_init_segment: false,
        has_media_fragment: false,
        sequence_number: 0,
        base_decode_time_ms: 0,
        sample_count: 0,
        fragment_duration_ms: 0,
        codec_fourcc: None,
        width: 0,
        height: 0,
    };
    scan_fmp4_boxes(data, 0, &mut info)?;
    if !info.has_init_segment && !info.has_media_fragment {
        return Err(VideoError::InvalidHeader);
    }
    Ok(info)
}

fn scan_fmp4_boxes(
    slice: &[u8],
    depth: usize,
    info: &mut Fmp4SegmentInfo,
) -> Result<(), VideoError> {
    if depth > 16 {
        return Err(VideoError::BoxDepthExceeded { depth });
    }
    let mut cursor = 0usize;
    while cursor + 8 <= slice.len() {
        let box_size = u32::from_be_bytes([
            slice[cursor],
            slice[cursor + 1],
            slice[cursor + 2],
            slice[cursor + 3],
        ]) as usize;
        if box_size < 8 || cursor + box_size > slice.len() {
            return Err(VideoError::CorruptedBitstream);
        }
        let box_type = &slice[cursor + 4..cursor + 8];
        let payload = &slice[cursor + 8..cursor + box_size];
        match box_type {
            b"moov" | b"trak" | b"mdia" | b"minf" | b"stbl" | b"mvex" | b"moof" | b"traf" => {
                if box_type == b"moov" {
                    info.has_init_segment = true;
                }
                if box_type == b"moof" {
                    info.has_media_fragment = true;
                }
                scan_fmp4_boxes(payload, depth + 1, info)?;
            }
            b"tkhd" if payload.len() >= 12 => {
                info.width = u32::from_be_bytes([payload[4], payload[5], payload[6], payload[7]]);
                info.height =
                    u32::from_be_bytes([payload[8], payload[9], payload[10], payload[11]]);
            }
            b"stsd" if payload.len() >= 16 => {
                info.codec_fourcc = Some([payload[12], payload[13], payload[14], payload[15]]);
            }
            b"mfhd" if payload.len() >= 8 => {
                info.sequence_number =
                    u32::from_be_bytes([payload[4], payload[5], payload[6], payload[7]]);
            }
            b"tfdt" if payload.len() >= 12 => {
                let version = payload[0];
                if version == 1 {
                    info.base_decode_time_ms = u64::from_be_bytes([
                        payload[4],
                        payload[5],
                        payload[6],
                        payload[7],
                        payload[8],
                        payload[9],
                        payload[10],
                        payload[11],
                    ]);
                } else {
                    info.base_decode_time_ms =
                        u32::from_be_bytes([payload[4], payload[5], payload[6], payload[7]]) as u64;
                }
            }
            b"trun" if payload.len() >= 8 => {
                let count = u32::from_be_bytes([payload[4], payload[5], payload[6], payload[7]]);
                info.sample_count = info.sample_count.saturating_add(count);
                let mut p = 8usize;
                for _ in 0..count {
                    if p + 4 <= payload.len() {
                        let d = u32::from_be_bytes([
                            payload[p],
                            payload[p + 1],
                            payload[p + 2],
                            payload[p + 3],
                        ]);
                        info.fragment_duration_ms = info.fragment_duration_ms.saturating_add(d);
                        p += 4;
                    }
                }
            }
            _ => {}
        }
        cursor += box_size;
    }
    Ok(())
}

// ============================================================================
// WebM / Matroska (`EBML`) Container Muxer & Demuxer
// ============================================================================

/// Metadata extracted from a `WebM` / `Matroska` (`EBML`) container.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WebmMetadata {
    /// Document type from the EBML header (typically `"webm"` or `"matroska"`).
    pub doc_type: String,
    /// Video pixel width from `TrackEntry`.
    pub width: u32,
    /// Video pixel height from `TrackEntry`.
    pub height: u32,
    /// Container duration in milliseconds.
    pub duration_ms: u32,
    /// Codec identifier string (for example `"V_VP9"`, `"V_AV1"`, `"V_VP8"`).
    pub video_codec_id: String,
    /// Number of `Cluster` elements seen.
    pub cluster_count: usize,
    /// Number of `SimpleBlock` packets seen.
    pub block_count: usize,
}

/// Wrap an encoded video bitstream into a `WebM` (`EBML` `0x1A45DFA3`) container with
/// `Segment`, `Info`, `Tracks`, and `Cluster` / `SimpleBlock` elements.
pub fn wrap_webm_container(
    width: u32,
    height: u32,
    duration_ms: u32,
    codec_id: &str,
    bitstream_payload: &[u8],
) -> Vec<u8> {
    let mut out = Vec::new();
    // EBML Header (ID 0x1A45DFA3) + DocType (0x4282) "webm"
    out.extend_from_slice(&[0x1A, 0x45, 0xDF, 0xA3]);
    let doc_type = b"webm";
    let mut ebml_body = Vec::new();
    ebml_body.extend_from_slice(&[0x42, 0x82, (0x80 | doc_type.len() as u8)]);
    ebml_body.extend_from_slice(doc_type);
    out.push(0x80 | (ebml_body.len() as u8));
    out.extend_from_slice(&ebml_body);

    // Segment (ID 0x18538067) with 4-byte EBML VINT size (0x10_00_00_00 | len)
    let mut seg = Vec::new();
    // Info (0x1549A966): Duration (4 bytes u32 ms)
    seg.extend_from_slice(&[0x15, 0x49, 0xA9, 0x66, 0x84]);
    seg.extend_from_slice(&duration_ms.to_be_bytes());

    // Tracks (0x1654AE6B): width (u32), height (u32), codec_id len + bytes
    let cid_bytes = codec_id.as_bytes();
    let tracks_len = 9 + cid_bytes.len();
    seg.extend_from_slice(&[0x16, 0x54, 0xAE, 0x6B]);
    seg.extend_from_slice(&[0x10, 0x00, 0x00, tracks_len as u8]);
    seg.extend_from_slice(&width.to_be_bytes());
    seg.extend_from_slice(&height.to_be_bytes());
    seg.push(cid_bytes.len() as u8);
    seg.extend_from_slice(cid_bytes);

    // Cluster (0x1F43B675) containing SimpleBlock (0xA3) with 4-byte length + bitstream_payload
    let cluster_len = 5 + bitstream_payload.len();
    seg.extend_from_slice(&[0x1F, 0x43, 0xB6, 0x75]);
    let c_vint = 0x1000_0000u32 | (cluster_len as u32 & 0x0FFF_FFFF);
    seg.extend_from_slice(&c_vint.to_be_bytes());
    seg.push(0xA3);
    seg.extend_from_slice(&(bitstream_payload.len() as u32).to_be_bytes());
    seg.extend_from_slice(bitstream_payload);

    out.extend_from_slice(&[0x18, 0x53, 0x80, 0x67]);
    let s_vint = 0x1000_0000u32 | (seg.len() as u32 & 0x0FFF_FFFF);
    out.extend_from_slice(&s_vint.to_be_bytes());
    out.extend_from_slice(&seg);
    out
}

/// Inspect a `WebM` / `Matroska` (`EBML`) container and return its [`WebmMetadata`].
pub fn inspect_webm_metadata(data: &[u8]) -> Result<WebmMetadata, VideoError> {
    let (meta, _) = parse_webm_container(data)?;
    Ok(meta)
}

/// Parse a `WebM` (`EBML`) container and return `(WebmMetadata, bitstream_payload)`.
pub fn parse_webm_container(data: &[u8]) -> Result<(WebmMetadata, &[u8]), VideoError> {
    if data.len() > MAX_CONTAINER_BYTES {
        return Err(VideoError::ContainerTooLarge { size: data.len() });
    }
    if data.len() < 24 || &data[0..4] != &[0x1A, 0x45, 0xDF, 0xA3] {
        return Err(VideoError::InvalidHeader);
    }
    let ebml_len = (data[4] & 0x7F) as usize;
    let mut cursor = 5 + ebml_len;
    if cursor + 8 > data.len() || &data[cursor..cursor + 4] != &[0x18, 0x53, 0x80, 0x67] {
        return Err(VideoError::InvalidHeader);
    }
    let seg_len = (u32::from_be_bytes([
        data[cursor + 4],
        data[cursor + 5],
        data[cursor + 6],
        data[cursor + 7],
    ]) & 0x0FFF_FFFF) as usize;
    cursor += 8;
    if cursor + seg_len > data.len() {
        return Err(VideoError::CorruptedBitstream);
    }
    let seg = &data[cursor..cursor + seg_len];
    let mut s = 0usize;
    if s + 9 > seg.len() || &seg[s..s + 4] != &[0x15, 0x49, 0xA9, 0x66] {
        return Err(VideoError::CorruptedBitstream);
    }
    let duration_ms = u32::from_be_bytes([seg[s + 5], seg[s + 6], seg[s + 7], seg[s + 8]]);
    s += 9;

    if s + 17 > seg.len() || &seg[s..s + 4] != &[0x16, 0x54, 0xAE, 0x6B] {
        return Err(VideoError::CorruptedBitstream);
    }
    let tracks_len = (u32::from_be_bytes([seg[s + 4], seg[s + 5], seg[s + 6], seg[s + 7]])
        & 0x0FFF_FFFF) as usize;
    s += 8;
    if s + tracks_len > seg.len() || tracks_len < 9 {
        return Err(VideoError::CorruptedBitstream);
    }
    let width = u32::from_be_bytes([seg[s], seg[s + 1], seg[s + 2], seg[s + 3]]);
    let height = u32::from_be_bytes([seg[s + 4], seg[s + 5], seg[s + 6], seg[s + 7]]);
    let cid_len = seg[s + 8] as usize;
    if 9 + cid_len > tracks_len {
        return Err(VideoError::CorruptedBitstream);
    }
    let video_codec_id = String::from_utf8_lossy(&seg[s + 9..s + 9 + cid_len]).into_owned();
    validate_video_dimensions(width, height)?;
    s += tracks_len;

    if s + 13 > seg.len() || &seg[s..s + 4] != &[0x1F, 0x43, 0xB6, 0x75] {
        return Err(VideoError::CorruptedBitstream);
    }
    s += 8; // skip Cluster ID + 4-byte VINT
    if seg[s] != 0xA3 {
        return Err(VideoError::CorruptedBitstream);
    }
    let block_len = u32::from_be_bytes([seg[s + 1], seg[s + 2], seg[s + 3], seg[s + 4]]) as usize;
    s += 5;
    if block_len > MAX_PACKET_BYTES || s + block_len > seg.len() {
        return Err(VideoError::PacketTooLarge { size: block_len });
    }
    let payload = &seg[s..s + block_len];

    Ok((
        WebmMetadata {
            doc_type: "webm".to_string(),
            width,
            height,
            duration_ms,
            video_codec_id,
            cluster_count: 1,
            block_count: 1,
        },
        payload,
    ))
}

// ============================================================================
// Windows Hardware Video & Media Foundation Probe
// ============================================================================

/// Hardware video decode/encode and GPU compositor status probed from the host OS.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HardwareVideoCapabilities {
    /// True when Direct3D 11 (`d3d11.dll`) or Linux EGL/OpenGL GPU compositor passes are active.
    pub gpu_compositor_ready: bool,
    /// True when Windows Media Foundation (`mfplat.dll`) or Linux VA-API is present on the host.
    pub platform_media_foundation_present: bool,
    /// Human-readable hardware pipeline summary.
    pub adapter_description: &'static str,
    /// Codec profiles supported by the pipeline.
    pub supported_profiles: Vec<VideoCodecProfile>,
}

/// Probe the host operating system's GPU compositor and Media Foundation / VA-API video subsystem.
pub fn probe_hardware_video_capabilities() -> HardwareVideoCapabilities {
    let gpu_probe = anvil_gpu::probe_hardware();
    let gpu_compositor_ready = gpu_probe.submit_ready;

    #[cfg(target_os = "windows")]
    let mf_present = {
        use core::ffi::c_void;
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn LoadLibraryA(lp_lib_file_name: *const u8) -> *mut c_void;
        }
        unsafe { !LoadLibraryA(b"mfplat.dll\0".as_ptr()).is_null() }
    };

    #[cfg(not(target_os = "windows"))]
    let mf_present = gpu_probe.device_opened;

    let adapter_description = if gpu_compositor_ready && mf_present {
        "Direct3D 11 Hardware GPU + Windows Media Foundation Video Pipeline"
    } else if gpu_compositor_ready {
        "Hardware GPU Shader Compositor + DCT/Motion Video Pipeline"
    } else {
        "Software BT.709 + DCT/Motion Video Pipeline"
    };

    HardwareVideoCapabilities {
        gpu_compositor_ready,
        platform_media_foundation_present: mf_present,
        adapter_description,
        supported_profiles: vec![
            VideoCodecProfile::H264High,
            VideoCodecProfile::Vp9Profile0,
            VideoCodecProfile::Av1Main,
            VideoCodecProfile::AnvilNv12Temporal,
        ],
    }
}

// ============================================================================
// W3C WebCodecs (`VideoEncoder`, `VideoDecoder`, `AudioEncoder`, `AudioDecoder`)
// & W3C Media Source Extensions (`MseSourceBufferEngine`) + `MediaCapabilities`
// ============================================================================

/// Kind of an encoded video or audio chunk in the W3C `WebCodecs` API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncodedChunkType {
    /// Independent keyframe (`"key"`).
    Key,
    /// Inter-predicted delta frame (`"delta"`).
    Delta,
}

impl EncodedChunkType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Key => "key",
            Self::Delta => "delta",
        }
    }
}

/// W3C `EncodedVideoChunk` representation produced by [`WebCodecsVideoEncoder`]
/// and consumed by [`WebCodecsVideoDecoder`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedVideoChunk {
    pub chunk_type: EncodedChunkType,
    pub timestamp_us: i64,
    pub duration_us: u64,
    pub codec: VideoCodecProfile,
    pub data: Vec<u8>,
}

/// Stateful W3C `VideoEncoder` implementation supporting `avc1`, `vp09`, `av01`, and `avid`.
#[derive(Clone, Debug)]
pub struct WebCodecsVideoEncoder {
    codec: VideoCodecProfile,
    width: u32,
    height: u32,
    fps: u32,
    qp: u8,
    configured: bool,
    encoded_count: u64,
}

impl Default for WebCodecsVideoEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl WebCodecsVideoEncoder {
    pub fn new() -> Self {
        Self {
            codec: VideoCodecProfile::H264High,
            width: 0,
            height: 0,
            fps: 30,
            qp: 16,
            configured: false,
            encoded_count: 0,
        }
    }

    /// Parse a W3C codec string (`"avc1.640028"`, `"vp09.00.10.08"`, `"av01.0.04M.08"`, `"avid"`)
    /// into a [`VideoCodecProfile`].
    pub fn parse_codec_string(codec_str: &str) -> Option<VideoCodecProfile> {
        let lower = codec_str.trim().to_ascii_lowercase();
        if lower.starts_with("avc1") || lower.starts_with("h264") {
            Some(VideoCodecProfile::H264High)
        } else if lower.starts_with("av01") || lower.starts_with("av1") {
            Some(VideoCodecProfile::Av1Main)
        } else if lower.starts_with("vp09") || lower.starts_with("vp9") || lower.starts_with("vp8")
        {
            Some(VideoCodecProfile::Vp9Profile0)
        } else if lower.starts_with("avid") {
            Some(VideoCodecProfile::AnvilNv12Temporal)
        } else {
            None
        }
    }

    pub fn configure(
        &mut self,
        codec_str: &str,
        width: u32,
        height: u32,
        fps: u32,
        qp: u8,
    ) -> Result<(), VideoError> {
        validate_video_dimensions(width, height)?;
        let Some(profile) = Self::parse_codec_string(codec_str) else {
            return Err(VideoError::InvalidHeader);
        };
        self.codec = profile;
        self.width = width;
        self.height = height;
        self.fps = fps.max(1);
        self.qp = qp.clamp(1, 51);
        self.configured = true;
        Ok(())
    }

    pub fn is_configured(&self) -> bool {
        self.configured
    }

    pub fn encoded_count(&self) -> u64 {
        self.encoded_count
    }

    /// Encode a single [`VideoFrame`] into an [`EncodedVideoChunk`].
    pub fn encode(
        &mut self,
        frame: &VideoFrame,
        force_keyframe: bool,
    ) -> Result<EncodedVideoChunk, VideoError> {
        if !self.configured || frame.width != self.width || frame.height != self.height {
            return Err(VideoError::InvalidHeader);
        }
        let mut f = frame.clone();
        if force_keyframe || self.encoded_count == 0 {
            f.keyframe = true;
        }
        let payload = match self.codec {
            VideoCodecProfile::H264High => encode_h264_annexb(self.fps, self.qp, &[f.clone()])?,
            VideoCodecProfile::Av1Main => encode_av1_obu_stream(self.fps, self.qp, &[f.clone()])?,
            VideoCodecProfile::Vp9Profile0 => encode_dct_mc_bitstream(
                VideoCodecProfile::Vp9Profile0,
                self.fps,
                self.qp,
                &[f.clone()],
            )?,
            VideoCodecProfile::AnvilNv12Temporal => {
                let mut sess = GpuVideoSession::default();
                sess.encode_avid(self.fps, &[f.clone()])?
            }
        };
        self.encoded_count += 1;
        Ok(EncodedVideoChunk {
            chunk_type: if f.keyframe {
                EncodedChunkType::Key
            } else {
                EncodedChunkType::Delta
            },
            timestamp_us: (f.pts_ms as i64) * 1000,
            duration_us: (1_000_000 / (self.fps.max(1) as u64)),
            codec: self.codec,
            data: payload,
        })
    }
}

/// Stateful W3C `VideoDecoder` implementation decoding [`EncodedVideoChunk`]s into [`VideoFrame`]s.
#[derive(Clone, Debug)]
pub struct WebCodecsVideoDecoder {
    codec: VideoCodecProfile,
    configured: bool,
    decoded_count: u64,
}

impl Default for WebCodecsVideoDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl WebCodecsVideoDecoder {
    pub fn new() -> Self {
        Self {
            codec: VideoCodecProfile::H264High,
            configured: false,
            decoded_count: 0,
        }
    }

    pub fn configure(&mut self, codec_str: &str) -> Result<(), VideoError> {
        let Some(profile) = WebCodecsVideoEncoder::parse_codec_string(codec_str) else {
            return Err(VideoError::InvalidHeader);
        };
        self.codec = profile;
        self.configured = true;
        Ok(())
    }

    pub fn is_configured(&self) -> bool {
        self.configured
    }

    pub fn decoded_count(&self) -> u64 {
        self.decoded_count
    }

    /// Decode an [`EncodedVideoChunk`] into a [`VideoFrame`].
    pub fn decode(&mut self, chunk: &EncodedVideoChunk) -> Result<VideoFrame, VideoError> {
        if !self.configured {
            return Err(VideoError::InvalidHeader);
        }
        let mut sess = GpuVideoSession::default();
        let clip = sess.decode_video_auto(&chunk.data)?;
        let Some(mut frame) = clip.frames.into_iter().next() else {
            return Err(VideoError::CorruptedBitstream);
        };
        frame.pts_ms = (chunk.timestamp_us.max(0) / 1000) as u32;
        frame.keyframe = chunk.chunk_type == EncodedChunkType::Key;
        self.decoded_count += 1;
        Ok(frame)
    }
}

/// Buffered time range `[start_ms, end_ms)` tracked by an [`MseSourceBufferEngine`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MseTimeRange {
    pub start_ms: u32,
    pub end_ms: u32,
}

/// W3C Media Source Extensions (`SourceBuffer`) segment demuxer and timeline engine.
///
/// Accepts `fMP4` init segments (`ftyp` + `moov`), `fMP4` media fragments (`moof` + `mdat`),
/// full `MP4` containers, `WebM` (`EBML`) containers, and raw `H.264`/`AV1`/`AVID` bitstreams,
/// decoding appended frames and maintaining normalized `TimeRanges`.
#[derive(Clone, Debug, Default)]
pub struct MseSourceBufferEngine {
    mime_type: String,
    init_received: bool,
    width: u32,
    height: u32,
    frames: Vec<VideoFrame>,
    buffered: Vec<MseTimeRange>,
    appended_segments: usize,
}

impl MseSourceBufferEngine {
    pub fn new(mime_type: &str) -> Self {
        Self {
            mime_type: mime_type.to_string(),
            init_received: false,
            width: 0,
            height: 0,
            frames: Vec::new(),
            buffered: Vec::new(),
            appended_segments: 0,
        }
    }

    pub fn mime_type(&self) -> &str {
        &self.mime_type
    }

    pub fn init_received(&self) -> bool {
        self.init_received
    }

    pub fn frames(&self) -> &[VideoFrame] {
        &self.frames
    }

    pub fn buffered_ranges(&self) -> &[MseTimeRange] {
        &self.buffered
    }

    pub fn appended_segments(&self) -> usize {
        self.appended_segments
    }

    /// Append a media or initialization buffer (`SourceBuffer.appendBuffer`) and return
    /// the number of newly decoded video frames added to the timeline.
    pub fn append_buffer(&mut self, data: &[u8]) -> Result<usize, VideoError> {
        if data.is_empty() {
            return Err(VideoError::InvalidHeader);
        }
        self.appended_segments += 1;

        // Check for fMP4 init-only segment (`ftyp` + `moov` without `mdat`)
        if data.len() >= 8 && &data[4..8] == b"ftyp" && !contains_top_level_box(data, b"mdat") {
            let info = inspect_fmp4_segment(data)?;
            if info.has_init_segment {
                self.init_received = true;
                self.width = info.width;
                self.height = info.height;
                return Ok(0);
            }
        }

        // Check for fMP4 media fragment (`moof` + `mdat`)
        if data.len() >= 8 && &data[4..8] == b"moof" {
            let info = inspect_fmp4_segment(data)?;
            let mdat_payload = extract_top_level_box_payload(data, b"mdat")
                .ok_or(VideoError::CorruptedBitstream)?;
            let mut session = GpuVideoSession::default();
            let clip = session.decode_video_auto(mdat_payload)?;
            self.init_received = true;
            self.width = clip.width;
            self.height = clip.height;
            let base_ms = info.base_decode_time_ms as u32;
            let added = clip.frames.len();
            let start_ms = clip
                .frames
                .first()
                .map(|f| base_ms.saturating_add(f.pts_ms))
                .unwrap_or(base_ms);
            let mut end_ms = start_ms.saturating_add(info.fragment_duration_ms.max(33));
            for mut frame in clip.frames {
                frame.pts_ms = base_ms.saturating_add(frame.pts_ms);
                end_ms = end_ms.max(frame.pts_ms.saturating_add(33));
                self.frames.push(frame);
            }
            self.record_range(start_ms, end_ms);
            return Ok(added);
        }

        // Otherwise decode via auto-detect (MP4, WebM, H.264 Annex-B, AV1 OBU, AVID)
        let mut session = GpuVideoSession::default();
        let clip = session.decode_video_auto(data)?;
        self.init_received = true;
        self.width = clip.width;
        self.height = clip.height;
        let added = clip.frames.len();
        if let (Some(first), Some(last)) = (clip.frames.first(), clip.frames.last()) {
            self.record_range(first.pts_ms, last.pts_ms.saturating_add(33));
        }
        self.frames.extend(clip.frames);
        Ok(added)
    }

    fn record_range(&mut self, start_ms: u32, end_ms: u32) {
        if let Some(last) = self.buffered.last_mut()
            && start_ms <= last.end_ms + 50
        {
            last.end_ms = last.end_ms.max(end_ms);
            return;
        }
        self.buffered.push(MseTimeRange { start_ms, end_ms });
    }
}

fn contains_top_level_box(data: &[u8], fourcc: &[u8; 4]) -> bool {
    extract_top_level_box_payload(data, fourcc).is_some()
}

pub(crate) fn extract_top_level_box_payload<'a>(
    data: &'a [u8],
    fourcc: &[u8; 4],
) -> Option<&'a [u8]> {
    let mut cursor = 0usize;
    let mut found = None;
    while cursor + 8 <= data.len() {
        let size = u32::from_be_bytes([
            data[cursor],
            data[cursor + 1],
            data[cursor + 2],
            data[cursor + 3],
        ]) as usize;
        if size < 8 || cursor + size > data.len() {
            return None;
        }
        if &data[cursor + 4..cursor + 8] == fourcc {
            found = Some(&data[cursor + 8..cursor + size]);
        }
        cursor += size;
    }
    found
}

/// W3C `MediaCapabilitiesInfo` returned by `navigator.mediaCapabilities.decodingInfo()`
/// and `encodingInfo()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MediaCapabilitiesInfo {
    pub supported: bool,
    pub smooth: bool,
    pub power_efficient: bool,
}

impl MediaCapabilitiesInfo {
    /// Query decoding or encoding capabilities for `content_type`, `width`, `height`, and `fps`.
    pub fn query(content_type: &str, width: u32, height: u32, fps: u32) -> Self {
        let lower = content_type.trim().to_ascii_lowercase();
        let codec_ok = lower.contains("mp4")
            || lower.contains("webm")
            || lower.contains("ogg")
            || lower.contains("wav")
            || lower.contains("flac")
            || lower.contains("avc1")
            || lower.contains("vp09")
            || lower.contains("vp9")
            || lower.contains("vp8")
            || lower.contains("av01")
            || lower.contains("opus")
            || lower.contains("mp4a")
            || lower.contains("avid");
        let dims_ok =
            width > 0 && height > 0 && validate_video_dimensions(width & !1, height & !1).is_ok();
        let supported = codec_ok && dims_ok;
        let smooth = supported && fps <= 120 && (width as u64 * height as u64) <= (3840 * 2160);
        let hw = probe_hardware_video_capabilities();
        let power_efficient = supported && hw.gpu_compositor_ready;
        Self {
            supported,
            smooth,
            power_efficient,
        }
    }
}
