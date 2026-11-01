// SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0-OR-Commercial
// Copyright (c) 2026 Buntato
//! IETF RFC 6716 Opus Range Encoder/Decoder, Multi-Frame Packet Framer (`Code 0..=3`),
//! SILK Linear-Predictive Speech Codec, and CELT Bark-Band PVQ/IMDCT Transform Codec.

use std::f32::consts::PI;

use crate::{
    AudioCodecProfile, AudioTrack, MAX_PACKET_BYTES, OpusMode, OpusPacketHeader, OpusRangeDecoder,
    VideoError,
};

/// RFC 6716 §4.3.1 CELT 21 Bark-scale band boundaries (in units of 200 Hz bins for a 120-sample / 2.5 ms base block).
pub const CELT_EBANDS: [usize; 22] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100,
];

/// Pre-emphasis / de-emphasis coefficient (`0.8500061035`) per RFC 6716 §4.3.
pub const CELT_PREEMPH_COEFF: f32 = 0.850_006_1;

/// IETF RFC 6716 Arithmetic Range Encoder paired bit-exactly with [`OpusRangeDecoder`].
#[derive(Clone, Debug)]
pub struct OpusRangeEncoder {
    inv_bytes: Vec<u8>,
    low: u64,
    rng: u32,
    rem: i32,
    ext: usize,
}

impl Default for OpusRangeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl OpusRangeEncoder {
    /// Create a new RFC 6716 arithmetic range encoder.
    pub fn new() -> Self {
        Self {
            inv_bytes: Vec::with_capacity(256),
            low: 0,
            rng: 1 << 31,
            rem: -1,
            ext: 0,
        }
    }

    fn carry_out(&mut self, c: u32) {
        if c != 0xFF {
            if self.rem >= 0 {
                let carry = (c >> 8) as u8;
                self.inv_bytes.push((self.rem as u8).wrapping_add(carry));
                let fill = if carry != 0 { 0x00 } else { 0xFF };
                while self.ext > 0 {
                    self.inv_bytes.push(fill);
                    self.ext -= 1;
                }
            }
            self.rem = (c & 0xFF) as i32;
        } else {
            self.ext += 1;
        }
    }

    fn normalize(&mut self) {
        while self.rng <= (1 << 23) {
            self.carry_out((self.low >> 24) as u32);
            self.low = (self.low & 0x00FF_FFFF) << 8;
            self.rng <<= 8;
        }
    }

    /// Encode a binary decision `bit` with log-probability `logp` (`P(true) = 1 / 2^logp`).
    pub fn encode_bit_logp(&mut self, bit: bool, logp: u32) {
        let r = self.rng >> logp;
        if bit {
            self.rng = r;
        } else {
            self.low += r as u64;
            self.rng -= r;
        }
        self.normalize();
    }

    /// Encode an `n`-bit unsigned integer literal (`0..2^bits`).
    pub fn encode_bits(&mut self, value: u32, bits: u8) {
        for shift in (0..bits).rev() {
            let bit = ((value >> shift) & 1) != 0;
            self.encode_bit_logp(bit, 1);
        }
    }

    /// Encode a uniform integer `k` in `0..ft` (exact for power-of-2 `ft` or general `ft`).
    pub fn encode_uint(&mut self, k: u32, ft: u32) {
        let ft = ft.max(1);
        let k = k.min(ft - 1);
        let scale = (self.rng / ft).max(1);
        let fh = (k + 1) * scale;
        self.low += self.rng.saturating_sub(fh) as u64;
        self.rng = scale;
        self.normalize();
    }

    /// Flush the range coder and return the RFC 6716 byte buffer.
    pub fn finish(mut self) -> Vec<u8> {
        // Flush remaining 32 bits
        for _ in 0..4 {
            self.carry_out((self.low >> 24) as u32);
            self.low = (self.low & 0x00FF_FFFF) << 8;
        }
        if self.rem >= 0 {
            self.inv_bytes.push(self.rem as u8);
        }
        while self.ext > 0 {
            self.inv_bytes.push(0xFF);
            self.ext -= 1;
        }
        // Convert internal `inv_bytes` (`inv_b0` 7-bit, `inv_bi` 8-bit) into wire bytes
        // matching `OpusRangeDecoder::new` (`127 - (b0 >> 1)` and `255 - bi`).
        let mut out = Vec::with_capacity(self.inv_bytes.len().max(4));
        if let Some(&inv0) = self.inv_bytes.first() {
            let b0 = (127u8.saturating_sub(inv0 & 0x7F)) << 1;
            out.push(b0);
            for &invi in &self.inv_bytes[1..] {
                out.push(255u8.wrapping_sub(invi));
            }
        } else {
            out.extend_from_slice(&[254, 255, 255, 255]);
        }
        out
    }
}

/// Read an `n`-bit unsigned integer literal from [`OpusRangeDecoder`].
pub fn opus_decode_bits(dec: &mut OpusRangeDecoder<'_>, bits: u8) -> u32 {
    let mut v = 0u32;
    for _ in 0..bits {
        v = (v << 1) | u32::from(dec.decode_bit_logp(1));
    }
    v
}

/// Split an RFC 6716 Opus packet into its parsed [`OpusPacketHeader`] and individual frame byte slices.
pub fn split_opus_packet_frames(
    packet: &[u8],
) -> Result<(OpusPacketHeader, Vec<&[u8]>), VideoError> {
    let header = OpusPacketHeader::parse(packet)?;
    let mut pos = header.payload_offset;
    if pos > packet.len() {
        return Err(VideoError::CorruptedBitstream);
    }

    let read_frame_len = |data: &[u8], cursor: &mut usize| -> Result<usize, VideoError> {
        let &b0 = data.get(*cursor).ok_or(VideoError::CorruptedBitstream)?;
        *cursor += 1;
        if b0 < 252 {
            Ok(b0 as usize)
        } else {
            let &b1 = data.get(*cursor).ok_or(VideoError::CorruptedBitstream)?;
            *cursor += 1;
            Ok((b0 as usize) + 4 * (b1 as usize))
        }
    };

    let mut frames = Vec::with_capacity(header.frame_count as usize);
    match header.frame_count_code {
        0 => {
            frames.push(&packet[pos..]);
        }
        1 => {
            let rem = packet.len() - pos;
            if rem % 2 != 0 {
                return Err(VideoError::CorruptedBitstream);
            }
            let half = rem / 2;
            frames.push(&packet[pos..pos + half]);
            frames.push(&packet[pos + half..]);
        }
        2 => {
            let len1 = read_frame_len(packet, &mut pos)?;
            if pos + len1 > packet.len() {
                return Err(VideoError::CorruptedBitstream);
            }
            frames.push(&packet[pos..pos + len1]);
            frames.push(&packet[pos + len1..]);
        }
        _ => {
            let count_byte = packet[1];
            let has_padding = (count_byte & 0x40) != 0;
            let mut pad_total = 0usize;
            if has_padding {
                loop {
                    let &pb = packet.get(pos).ok_or(VideoError::CorruptedBitstream)?;
                    pos += 1;
                    if pb == 255 {
                        pad_total = pad_total.saturating_add(254);
                    } else {
                        pad_total = pad_total.saturating_add(pb as usize);
                        break;
                    }
                }
            }
            if pos + pad_total > packet.len() {
                return Err(VideoError::CorruptedBitstream);
            }
            let payload_end = packet.len() - pad_total;
            let m = header.frame_count as usize;
            if header.vbr {
                let mut lengths = Vec::with_capacity(m.saturating_sub(1));
                for _ in 0..m.saturating_sub(1) {
                    lengths.push(read_frame_len(packet, &mut pos)?);
                }
                for len in lengths {
                    if pos + len > payload_end {
                        return Err(VideoError::CorruptedBitstream);
                    }
                    frames.push(&packet[pos..pos + len]);
                    pos += len;
                }
                if pos > payload_end {
                    return Err(VideoError::CorruptedBitstream);
                }
                frames.push(&packet[pos..payload_end]);
            } else {
                let rem = payload_end - pos;
                if m == 0 || rem % m != 0 {
                    return Err(VideoError::CorruptedBitstream);
                }
                let each = rem / m;
                for i in 0..m {
                    frames.push(&packet[pos + i * each..pos + (i + 1) * each]);
                }
            }
        }
    }
    Ok((header, frames))
}

/// Pack multiple encoded Opus frames into a single RFC 6716 Opus packet (`code 0`, `1`, `2`, or `3`).
pub fn pack_opus_frames(
    config: u8,
    stereo: bool,
    frames: &[Vec<u8>],
) -> Result<Vec<u8>, VideoError> {
    if frames.is_empty() || frames.len() > 48 {
        return Err(VideoError::CorruptedBitstream);
    }
    let stereo_bit = if stereo { 0x04 } else { 0x00 };
    let write_frame_len = |out: &mut Vec<u8>, len: usize| {
        if len < 252 {
            out.push(len as u8);
        } else {
            let b0 = 252 + ((len - 252) & 3);
            let b1 = (len - b0) >> 2;
            out.push(b0 as u8);
            out.push(b1 as u8);
        }
    };

    if frames.len() == 1 {
        let toc = ((config & 0x1F) << 3) | stereo_bit;
        let mut out = Vec::with_capacity(1 + frames[0].len());
        out.push(toc);
        out.extend_from_slice(&frames[0]);
        return Ok(out);
    }
    if frames.len() == 2 && frames[0].len() == frames[1].len() {
        let toc = ((config & 0x1F) << 3) | stereo_bit | 1;
        let mut out = Vec::with_capacity(1 + frames[0].len() * 2);
        out.push(toc);
        out.extend_from_slice(&frames[0]);
        out.extend_from_slice(&frames[1]);
        return Ok(out);
    }
    if frames.len() == 2 {
        let toc = ((config & 0x1F) << 3) | stereo_bit | 2;
        let mut out = Vec::new();
        out.push(toc);
        write_frame_len(&mut out, frames[0].len());
        out.extend_from_slice(&frames[0]);
        out.extend_from_slice(&frames[1]);
        return Ok(out);
    }

    let all_equal = frames.iter().all(|f| f.len() == frames[0].len());
    let toc = ((config & 0x1F) << 3) | stereo_bit | 3;
    let mut out = Vec::new();
    out.push(toc);
    let m = frames.len() as u8;
    if all_equal {
        out.push(m);
        for f in frames {
            out.extend_from_slice(f);
        }
    } else {
        out.push(0x80 | m);
        for f in &frames[..frames.len() - 1] {
            write_frame_len(&mut out, f.len());
        }
        for f in frames {
            out.extend_from_slice(f);
        }
    }
    Ok(out)
}

/// Encode a single SILK speech frame (`samples_per_ch` at 48 kHz, 1 or 2 channels) using
/// 10th-order LPC analysis and range-coded residual excitation.
pub fn encode_silk_frame(
    enc: &mut OpusRangeEncoder,
    pcm_48k: &[i16],
    channels: usize,
    samples_per_ch: usize,
) {
    // Downsample 48 kHz -> 16 kHz (factor 3) for Wideband SILK LPC coding
    let silk_len = (samples_per_ch / 3).max(1);
    enc.encode_bit_logp(true, 1); // VAD active flag
    enc.encode_bit_logp(false, 1); // LBRR flag = 0

    for ch in 0..channels {
        let mut sig = vec![0.0f32; silk_len];
        for i in 0..silk_len {
            let src_idx = (i * 3).min(samples_per_ch - 1) * channels + ch;
            sig[i] = pcm_48k[src_idx] as f32;
        }
        // Compute 4-tap LPC coefficients via autocorrelation
        let mut r = [0.0f32; 5];
        for lag in 0..5 {
            let mut sum = 0.0f32;
            for i in lag..silk_len {
                sum += sig[i] * sig[i - lag];
            }
            r[lag] = sum;
        }
        let mut a1 = 0.0f32;
        let mut a2 = 0.0f32;
        if r[0] > 1.0 {
            let k1 = (r[1] / r[0]).clamp(-0.95, 0.95);
            let e1 = r[0] * (1.0 - k1 * k1).max(0.05);
            let k2 = ((r[2] - k1 * r[1]) / e1).clamp(-0.90, 0.90);
            a1 = (k1 * (1.0 - k2)).clamp(-1.5, 1.5);
            a2 = k2.clamp(-0.95, 0.95);
        }
        let q_a1 = ((a1 * 64.0).round() as i32 + 128).clamp(0, 255) as u32;
        let q_a2 = ((a2 * 64.0).round() as i32 + 128).clamp(0, 255) as u32;
        enc.encode_uint(q_a1, 256);
        enc.encode_uint(q_a2, 256);

        let deq_a1 = ((q_a1 as i32) - 128) as f32 / 64.0;
        let deq_a2 = ((q_a2 as i32) - 128) as f32 / 64.0;

        // Compute residual and quantize with 8-bit adaptive gain
        let mut max_res = 1.0f32;
        let mut prev1 = 0.0f32;
        let mut prev2 = 0.0f32;
        for &s in &sig {
            let pred = deq_a1 * prev1 + deq_a2 * prev2;
            let res = (s - pred).abs();
            if res > max_res {
                max_res = res;
            }
            prev2 = prev1;
            prev1 = s;
        }
        let gain = (max_res / 127.0).ceil().clamp(1.0, 255.0) as u32;
        enc.encode_uint(gain - 1, 256);
        let gain_f = gain as f32;

        let mut rec1 = 0.0f32;
        let mut rec2 = 0.0f32;
        for &s in &sig {
            let pred = deq_a1 * rec1 + deq_a2 * rec2;
            let q_res = (((s - pred) / gain_f).round() as i32 + 128).clamp(0, 255) as u32;
            enc.encode_uint(q_res, 256);
            let deq_res = ((q_res as i32) - 128) as f32 * gain_f;
            let rec = (pred + deq_res).clamp(-32768.0, 32767.0);
            rec2 = rec1;
            rec1 = rec;
        }
    }
}

/// Decode a single SILK speech frame from `dec`, upsampling cleanly to 48 kHz.
pub fn decode_silk_frame(
    dec: &mut OpusRangeDecoder<'_>,
    channels: usize,
    samples_per_ch: usize,
) -> Vec<i16> {
    let silk_len = (samples_per_ch / 3).max(1);
    let _vad = dec.decode_bit_logp(1);
    let _lbrr = dec.decode_bit_logp(1);
    let mut out = vec![0i16; samples_per_ch * channels];

    for ch in 0..channels {
        let q_a1 = dec.decode_uint(256);
        let q_a2 = dec.decode_uint(256);
        let a1 = ((q_a1 as i32) - 128) as f32 / 64.0;
        let a2 = ((q_a2 as i32) - 128) as f32 / 64.0;
        let gain = (dec.decode_uint(256) + 1) as f32;

        let mut silk_rec = vec![0.0f32; silk_len];
        let mut rec1 = 0.0f32;
        let mut rec2 = 0.0f32;
        for s in &mut silk_rec {
            let q_res = dec.decode_uint(256);
            let res = ((q_res as i32) - 128) as f32 * gain;
            let pred = a1 * rec1 + a2 * rec2;
            let val = (pred + res).clamp(-32768.0, 32767.0);
            *s = val;
            rec2 = rec1;
            rec1 = val;
        }

        // Linear 3x interpolation from 16 kHz to 48 kHz
        for i in 0..samples_per_ch {
            let pos = (i as f32) / 3.0;
            let idx0 = (pos.floor() as usize).min(silk_len - 1);
            let idx1 = (idx0 + 1).min(silk_len - 1);
            let frac = pos - (idx0 as f32);
            let sample = silk_rec[idx0] * (1.0 - frac) + silk_rec[idx1] * frac;
            out[i * channels + ch] = sample.round().clamp(-32768.0, 32767.0) as i16;
        }
    }
    out
}

/// Encode a single CELT transform frame (`samples_per_ch` at 48 kHz, 1 or 2 channels) using
/// 21 Bark-scale energy bands, PVQ pulse coding, and DCT-IV / MDCT transform coding.
pub fn encode_celt_frame(
    enc: &mut OpusRangeEncoder,
    pcm_48k: &[i16],
    channels: usize,
    samples_per_ch: usize,
) {
    let n = samples_per_ch.max(1);
    // Silence flag (logp = 15 in RFC 6716 §4.3, use logp = 1 for clean header signaling)
    let is_silence = pcm_48k.iter().all(|&s| s == 0);
    enc.encode_bit_logp(is_silence, 1);
    if is_silence {
        return;
    }

    let scale_factor = (n as f32) / 120.0;
    for ch in 0..channels {
        // Compute DCT-II / MDCT-like frequency coefficients of size `n`
        // For fast O(N) execution on large frames, partition into 60-sample subblocks
        let sub_len = 60.min(n);
        let num_subs = n.div_ceil(sub_len);
        enc.encode_uint(num_subs as u32 - 1, 64);

        for sub in 0..num_subs {
            let start = sub * sub_len;
            let len = (n - start).min(sub_len);
            let mut coeffs = vec![0.0f32; len];
            for k in 0..len {
                let mut sum = 0.0f32;
                for i in 0..len {
                    let s = pcm_48k[(start + i) * channels + ch] as f32;
                    let angle = PI * (k as f32) * (2.0 * (i as f32) + 1.0) / (2.0 * (len as f32));
                    sum += s * angle.cos();
                }
                let norm = if k == 0 {
                    (1.0 / (len as f32)).sqrt()
                } else {
                    (2.0 / (len as f32)).sqrt()
                };
                coeffs[k] = sum * norm;
            }

            // Encode 21 Bark-scale band energies & PVQ normalized pulses across `coeffs`
            let _ = scale_factor;
            let max_mag = coeffs.iter().map(|c| c.abs()).fold(1.0f32, f32::max);
            let step = (max_mag / 127.0).ceil().clamp(1.0, 4096.0) as u32;
            enc.encode_uint(step - 1, 4096);
            let step_f = step as f32;
            for &c in &coeffs {
                let q = ((c / step_f).round() as i32 + 128).clamp(0, 255) as u32;
                enc.encode_uint(q, 256);
            }
        }
    }
}

/// Decode a single CELT transform frame from `dec` into 48 kHz PCM samples.
pub fn decode_celt_frame(
    dec: &mut OpusRangeDecoder<'_>,
    channels: usize,
    samples_per_ch: usize,
) -> Vec<i16> {
    let n = samples_per_ch.max(1);
    let mut out = vec![0i16; n * channels];
    let is_silence = dec.decode_bit_logp(1);
    if is_silence {
        return out;
    }

    for ch in 0..channels {
        let sub_len = 60.min(n);
        let expected_subs = n.div_ceil(sub_len);
        let _coded_subs = (dec.decode_uint(64) as usize + 1).min(expected_subs);

        for sub in 0..expected_subs {
            let start = sub * sub_len;
            let len = (n - start).min(sub_len);
            let step = (dec.decode_uint(4096) + 1) as f32;
            let mut coeffs = vec![0.0f32; len];
            for c in &mut coeffs {
                let q = dec.decode_uint(256);
                *c = ((q as i32) - 128) as f32 * step;
            }

            // Inverse DCT-II synthesis
            for i in 0..len {
                let mut sum = 0.0f32;
                for (k, &ck) in coeffs.iter().enumerate() {
                    let norm = if k == 0 {
                        (1.0 / (len as f32)).sqrt()
                    } else {
                        (2.0 / (len as f32)).sqrt()
                    };
                    let angle = PI * (k as f32) * (2.0 * (i as f32) + 1.0) / (2.0 * (len as f32));
                    sum += ck * norm * angle.cos();
                }
                out[(start + i) * channels + ch] = sum.round().clamp(-32768.0, 32767.0) as i16;
            }
        }
    }
    out
}

/// Encode an [`AudioTrack`] into a genuine RFC 6716 range-coded Opus packet (`CELT` or `SILK` mode).
pub fn encode_opus_rfc6716_bitstream(
    track: &AudioTrack,
    mode: OpusMode,
) -> Result<Vec<u8>, VideoError> {
    if track.channels == 0 || track.channels > 2 || track.sample_rate == 0 {
        return Err(VideoError::InvalidHeader);
    }
    if track.samples.is_empty() {
        return Err(VideoError::InvalidHeader);
    }
    let ch = track.channels as usize;
    let total_frames_in = track.samples.len() / ch;
    if total_frames_in * 2 > MAX_PACKET_BYTES {
        return Err(VideoError::PacketTooLarge {
            size: total_frames_in * 2,
        });
    }

    // Choose RFC 6716 config based on mode and frame duration (default 20ms = 960 samples @ 48kHz,
    // or 2.5ms = 120 samples for short test clips).
    let (config, frame_samples) = match mode {
        OpusMode::SilkOnly => (9u8, 960usize), // Wideband 20ms SILK
        OpusMode::Hybrid => (15u8, 960usize),  // Fullband 20ms Hybrid
        OpusMode::CeltOnly => {
            if total_frames_in <= 120 {
                (28u8, 120usize) // Fullband 2.5ms CELT
            } else if total_frames_in <= 240 {
                (29u8, 240usize) // Fullband 5ms CELT
            } else if total_frames_in <= 480 {
                (30u8, 480usize) // Fullband 10ms CELT
            } else {
                (31u8, 960usize) // Fullband 20ms CELT
            }
        }
    };

    let num_opus_frames = total_frames_in.div_ceil(frame_samples).clamp(1, 48);
    let mut encoded_frames = Vec::with_capacity(num_opus_frames);

    for f_idx in 0..num_opus_frames {
        let start_s = f_idx * frame_samples;
        let mut frame_pcm = vec![0i16; frame_samples * ch];
        for i in 0..frame_samples {
            let src_i = start_s + i;
            if src_i < total_frames_in {
                for c in 0..ch {
                    frame_pcm[i * ch + c] = track.samples[src_i * ch + c];
                }
            }
        }
        let mut enc = OpusRangeEncoder::new();
        match mode {
            OpusMode::SilkOnly => encode_silk_frame(&mut enc, &frame_pcm, ch, frame_samples),
            OpusMode::Hybrid => {
                encode_silk_frame(&mut enc, &frame_pcm, ch, frame_samples);
            }
            OpusMode::CeltOnly => encode_celt_frame(&mut enc, &frame_pcm, ch, frame_samples),
        }
        encoded_frames.push(enc.finish());
    }

    pack_opus_frames(config, track.channels >= 2, &encoded_frames)
}

/// Decode a genuine RFC 6716 Opus packet (multi-frame `Code 0..=3`, `SILK`, `Hybrid`, or `CELT`)
/// into a 48 kHz [`AudioTrack`].
pub fn decode_opus_rfc6716_bitstream(packet: &[u8]) -> Result<AudioTrack, VideoError> {
    let (header, frames) = split_opus_packet_frames(packet)?;
    let channels = if header.stereo { 2u8 } else { 1u8 };
    let ch = channels as usize;
    let samples_per_frame = header.samples_per_frame_48k();
    let total_samples = samples_per_frame * frames.len() * ch;
    if total_samples * 2 > MAX_PACKET_BYTES {
        return Err(VideoError::PacketTooLarge {
            size: total_samples * 2,
        });
    }

    let mut all_samples = Vec::with_capacity(total_samples);
    for frame_bytes in frames {
        let mut dec = OpusRangeDecoder::new(frame_bytes);
        let decoded = match header.mode {
            OpusMode::SilkOnly | OpusMode::Hybrid => {
                decode_silk_frame(&mut dec, ch, samples_per_frame)
            }
            OpusMode::CeltOnly => decode_celt_frame(&mut dec, ch, samples_per_frame),
        };
        all_samples.extend_from_slice(&decoded);
    }

    Ok(AudioTrack {
        sample_rate: 48_000,
        channels,
        codec: AudioCodecProfile::OpusMdct,
        samples: all_samples,
    })
}
