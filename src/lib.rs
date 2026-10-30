// SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0-OR-Commercial
// Copyright (c) 2026 Buntato
//! CPU RGBA frames in a small ISOBMFF container, with optional AES-128-CTR on the sample.
//!
//! `wrap_mp4_container` writes three top-level boxes: `ftyp`, `moov`, and `mdat`.
//! `moov` holds `mvhd` and `trak`. `trak` holds `tkhd` only.
//! `wrap_encrypted_mp4_container` writes those same boxes and also puts `sinf`
//! (`schm` and `tenc`) and `pssh` inside `moov`. The `mdat` payload is then one
//! ciphertext sample. Readers require a leading `ftyp` plus a `moov` and an `mdat`
//! somewhere in the walk. The walk also descends into `trak`, `mdia`, `minf`,
//! `stbl`, and `sinf` when they are present, and it copies fields from `mvhd`,
//! `tkhd`, `schm`, `tenc`, and `pssh`. Each box size is a big-endian `u32` that
//! includes the 8-byte header. Nesting past [`MAX_MP4_BOX_DEPTH`] is rejected.
//! The written `mvhd` and `tkhd` are short layouts from this file, not full ISO headers.
//!
//! A [`VideoFrame`] is packed `RGBA8888` (4 bytes per pixel). Encode calls
//! `Nv12Surface::from_rgba_bt709` and stores a Y plane plus an interleaved UV plane,
//! or a wrapping byte residual of those planes against the previous frame.
//! `compress_zlib` then wraps that payload. Those packets form an `AVID` bitstream,
//! and that bitstream is the `mdat` body. Decode calls `decompress_zlib_bounded`
//! and `Nv12Surface::to_rgba_bt709`.
//!
//! Sample encryption uses a 16-byte AES-128 key and a 16-byte IV, which is the
//! initial CTR counter. [`cenc_encrypt_sample`] and [`cenc_decrypt_sample`] both
//! call `aes128_ctr`. The whole `AVID` buffer is one sample. There are no per-frame
//! subsample ranges.
//!
//! Three sibling crates are called, and only for these calls:
//! - `anvil-compress`: `compress_zlib` on encode, `decompress_zlib_bounded` on decode.
//! - `anvil-cipher`: `aes128_ctr` from both sample functions above.
//! - `anvil-gpu`: `convert_rgba_on_device` and `present_nv12_on_device` when the
//!   hardware flag is set. Those calls shade through the tile device. The flag-off
//!   path still uses `Nv12Surface` directly.
//!
//! Not implemented: a hardware decoder, a GPU decode or encode session, H.264, AV1,
//! and a full sample-encryption layout (no `senc`, `saiz`, `saio`, or subsample
//! pattern). [`GpuVideoSession`] stores a backend label, two flags, two counters,
//! and an in-memory demo key list. It does not open a device. A video element is
//! not parsed here.
//!
//! Limits checked in this file: [`MAX_VIDEO_WIDTH`] 4096, [`MAX_VIDEO_HEIGHT`] 2160,
//! [`MAX_VIDEO_PIXELS`], [`MAX_VIDEO_FRAMES`] 36000, [`MAX_PACKET_BYTES`] 16 MiB,
//! [`MAX_CONTAINER_BYTES`] 64 MiB, [`MAX_MP4_BOXES`] 256, [`MAX_MP4_BOX_DEPTH`] 16.

use std::fmt;

use anvil_cipher::aes128_ctr;
use anvil_compress::{CompressError, compress_zlib, decompress_zlib_bounded};
use anvil_gpu::{Nv12Surface, convert_rgba_on_device, present_nv12_on_device};

pub mod codecs;
pub use codecs::*;

/// Largest accepted frame width, in pixels.
pub const MAX_VIDEO_WIDTH: u32 = 4096;

/// Largest accepted frame height, in pixels.
pub const MAX_VIDEO_HEIGHT: u32 = 2160;

/// Largest accepted pixel count for one frame (`4096 * 2160`).
pub const MAX_VIDEO_PIXELS: u64 = 8_847_360;

/// Largest accepted frame count in one `AVID` bitstream.
pub const MAX_VIDEO_FRAMES: u32 = 36_000;

/// Largest zlib packet, and the largest `mdat` payload, in bytes (16 MiB).
pub const MAX_PACKET_BYTES: usize = 16 * 1024 * 1024;

/// Largest MP4 byte length accepted before boxes are read (64 MiB).
pub const MAX_CONTAINER_BYTES: usize = 64 * 1024 * 1024;

/// Largest number of boxes one container walk will count.
pub const MAX_MP4_BOXES: usize = 256;

/// Largest `depth` the box walk accepts. The file itself is scanned at depth 0.
pub const MAX_MP4_BOX_DEPTH: usize = 16;

/// 16-byte system id copied into `pssh`.
///
/// The bytes are `1077efec-c0b2-4d02-ace3-3c1e52e2fb4b`. This is a fixed array in
/// this file, not a value loaded from disk. Writing it does not perform a key-system handshake.
pub const W3C_CLEARKEY_SYSTEM_ID: [u8; 16] = [
    0x10, 0x77, 0xef, 0xec, 0xc0, 0xb2, 0x4d, 0x02, 0xac, 0xe3, 0x3c, 0x1e, 0x52, 0xe2, 0xfb, 0x4b,
];

/// 16-byte demo key id written into `tenc` and `pssh` by the encrypted demo.
///
/// Fixed array in this file. Not a secret fetched from disk.
// Fixed demo constant. Not a secret fetched from disk.
pub const DEFAULT_EME_DEMO_KID: [u8; 16] = [
    0x41, 0x4e, 0x56, 0x4c, 0x5f, 0x43, 0x45, 0x4e, 0x43, 0x5f, 0x4b, 0x49, 0x44, 0x5f, 0x30, 0x31,
];

/// 16-byte demo AES-128 key used by the encrypted demo.
///
/// Key size is 16 bytes. Callers pass a separate 16-byte IV as the CTR counter.
/// [`cenc_encrypt_sample`] and [`cenc_decrypt_sample`] both hand this key to `aes128_ctr`.
/// Fixed array in this file. Not a secret fetched from disk.
// Fixed demo constant. Not a secret fetched from disk.
pub const DEFAULT_EME_DEMO_KEY: [u8; 16] = [
    0x9a, 0x7b, 0x3c, 0x5d, 0x1e, 0x8f, 0x20, 0x41, 0x62, 0x83, 0xa4, 0xc5, 0xe6, 0x07, 0x18, 0x29,
];

/// Failure from a limit check, a box walk, an `AVID` parse, or zlib.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VideoError {
    /// Magic, version, empty input, pixel length, or a required box was wrong or missing.
    InvalidHeader,
    /// Width, height, or pixel count is above the limits in this file.
    DimensionsTooLarge { width: u32, height: u32 },
    /// Width or height is not even. The NV12 conversion needs both even.
    OddDimensions { width: u32, height: u32 },
    /// Declared frame count is 0 or above [`MAX_VIDEO_FRAMES`].
    FrameCountTooLarge { count: u32 },
    /// A zlib packet or an `mdat` payload is above [`MAX_PACKET_BYTES`].
    PacketTooLarge { size: usize },
    /// The file is longer than [`MAX_CONTAINER_BYTES`].
    ContainerTooLarge { size: usize },
    /// The walk counted more boxes than [`MAX_MP4_BOXES`].
    TooManyBoxes { count: usize },
    /// Nested depth is greater than [`MAX_MP4_BOX_DEPTH`].
    BoxDepthExceeded { depth: usize },
    /// A box size, frame type, or plane length does not match the buffer.
    CorruptedBitstream,
    /// The 16-byte key id is not in the in-memory demo key list.
    MissingDecryptionKey { kid: [u8; 16] },
    /// `anvil-compress` rejected the zlib packet.
    Compression(CompressError),
}

impl fmt::Display for VideoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VideoError::InvalidHeader => write!(f, "invalid video stream or container header"),
            VideoError::DimensionsTooLarge { width, height } => write!(
                f,
                "video dimensions {width}x{height} exceed 4K security cap ({MAX_VIDEO_WIDTH}x{MAX_VIDEO_HEIGHT})"
            ),
            VideoError::OddDimensions { width, height } => write!(
                f,
                "video dimensions {width}x{height} must be even for 4:2:0 NV12 surfaces"
            ),
            VideoError::FrameCountTooLarge { count } => {
                write!(
                    f,
                    "video frame count {count} exceeds security limit ({MAX_VIDEO_FRAMES})"
                )
            }
            VideoError::PacketTooLarge { size } => {
                write!(
                    f,
                    "video packet size {size} exceeds security limit ({MAX_PACKET_BYTES})"
                )
            }
            VideoError::ContainerTooLarge { size } => {
                write!(
                    f,
                    "video container size {size} exceeds security limit ({MAX_CONTAINER_BYTES})"
                )
            }
            VideoError::TooManyBoxes { count } => {
                write!(
                    f,
                    "MP4 container box count {count} exceeds security limit ({MAX_MP4_BOXES})"
                )
            }
            VideoError::BoxDepthExceeded { depth } => {
                write!(
                    f,
                    "MP4 container box nesting depth {depth} exceeds security limit ({MAX_MP4_BOX_DEPTH})"
                )
            }
            VideoError::CorruptedBitstream => write!(f, "corrupted video bitstream"),
            VideoError::MissingDecryptionKey { kid } => {
                write!(
                    f,
                    "missing EME/CDM decryption key for key ID {:02x}{:02x}{:02x}{:02x}",
                    kid[0], kid[1], kid[2], kid[3]
                )
            }
            VideoError::Compression(e) => write!(f, "video compression error: {e}"),
        }
    }
}

impl std::error::Error for VideoError {}

impl From<CompressError> for VideoError {
    /// Wrap a zlib error as [`VideoError::Compression`].
    fn from(value: CompressError) -> Self {
        VideoError::Compression(value)
    }
}

/// Name stored on a clip. This crate does not open a GPU device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuVideoBackend {
    /// Label chosen on Windows when the hardware flag is set. No Direct3D calls.
    D3d11Dxva2,
    /// Label chosen on Linux when the hardware flag is set. No VA-API calls.
    LinuxVaApi,
    /// Label used when the hardware flag is clear, and on other operating systems.
    /// Frame bytes are converted on the CPU in every case.
    SoftwareFallback,
}

impl GpuVideoBackend {
    /// Static text for this label. The text is not evidence that a device was opened.
    pub fn label(self) -> &'static str {
        match self {
            GpuVideoBackend::D3d11Dxva2 => "D3D11 Video / DXVA2 (NV12 GPU)",
            GpuVideoBackend::LinuxVaApi => "VA-API / DRM (NV12 GPU)",
            GpuVideoBackend::SoftwareFallback => "Software BT.709 Pipeline",
        }
    }

    /// `true` for the Windows and Linux labels, `false` for [`GpuVideoBackend::SoftwareFallback`].
    ///
    /// This matches on the enum only. It does not query a device.
    pub fn is_hardware_accelerated(self) -> bool {
        !matches!(self, GpuVideoBackend::SoftwareFallback)
    }
}

/// Four-byte codec profile identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoCodecProfile {
    /// H.264 / AVC1 (`avc1`) profile backed by Annex-B NAL unit and DCT/MC bitstream coding.
    H264High,
    /// AV1 (`av01`) profile backed by OBU and DCT/MC bitstream coding.
    Av1Main,
    /// VP9 (`vp09`) profile backed by OBU/WebM and DCT/MC bitstream coding.
    Vp9Profile0,
    /// VP8 (`vp08`) profile backed by RFC 6386 keyframe/interframe and WHT/DCT/MC bitstream coding.
    Vp8Profile0,
    /// Label for the `AVID` bitstream this crate reads and writes. [`Self::fourcc`] is `avid`.
    AnvilNv12Temporal,
}

impl VideoCodecProfile {
    /// `avc1`, `av01`, `vp09`, `vp08`, or `avid`.
    pub fn fourcc(self) -> [u8; 4] {
        match self {
            VideoCodecProfile::H264High => *b"avc1",
            VideoCodecProfile::Av1Main => *b"av01",
            VideoCodecProfile::Vp9Profile0 => *b"vp09",
            VideoCodecProfile::Vp8Profile0 => *b"vp08",
            VideoCodecProfile::AnvilNv12Temporal => *b"avid",
        }
    }
}

const IMA_INDEX_TABLE: [i32; 16] = [-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8];

const IMA_STEP_TABLE: [i32; 89] = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66,
    73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449,
    494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272,
    2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493,
    10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
];

/// Audio codec profile used inside an `AAUD` stream or standalone `fLaC` / `AMDC` container.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioCodecProfile {
    /// Uncompressed 16-bit signed little-endian PCM wrapped in zlib.
    Pcm16Le,
    /// 4-bit IMA-ADPCM adaptive differential pulse-code modulation wrapped in zlib.
    ImaAdpcm4,
    /// Lossless FLAC (`fLaC`) linear prediction + Rice residual coding.
    FlacLossless,
    /// This crate's own 50%-overlapped MDCT frames. Not real Opus or AAC.
    OpusMdct,
}

/// Decoded interleaved 16-bit signed PCM audio track synchronized with a [`VideoClip`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioTrack {
    /// Sample rate in Hz (for example `44_100` or `48_000`).
    pub sample_rate: u32,
    /// Channel count (`1` for mono, `2` for stereo).
    pub channels: u8,
    /// Codec profile used when serializing to `AAUD`.
    pub codec: AudioCodecProfile,
    /// Interleaved 16-bit signed PCM samples (`L, R, L, R, ...` when stereo).
    pub samples: Vec<i16>,
}

impl AudioTrack {
    /// Total duration of this audio track in milliseconds.
    pub fn duration_ms(&self) -> u32 {
        let ch = (self.channels.max(1)) as u64;
        let sr = (self.sample_rate.max(1)) as u64;
        let frames = (self.samples.len() as u64) / ch;
        ((frames * 1000) / sr) as u32
    }

    /// Compute normalized peak audio level (`0..=100`) around `pts_ms` over `window_ms`.
    pub fn peak_level_at_ms(&self, pts_ms: u32, window_ms: u32) -> u8 {
        if self.samples.is_empty() || self.sample_rate == 0 || self.channels == 0 {
            return 0;
        }
        let ch = self.channels as usize;
        let sr = self.sample_rate as usize;
        let start_frame = ((pts_ms as usize) * sr) / 1000;
        let span_frames = (((window_ms.max(10) as usize) * sr) / 1000).max(1);
        let start_idx = (start_frame * ch).min(self.samples.len());
        let end_idx = ((start_frame + span_frames) * ch).min(self.samples.len());
        if start_idx >= end_idx {
            return 0;
        }
        let mut peak = 0i32;
        for &s in &self.samples[start_idx..end_idx] {
            let amp = (s as i32).abs();
            if amp > peak {
                peak = amp;
            }
        }
        ((peak * 100) / 32767).clamp(0, 100) as u8
    }

    /// Synthesize a harmonic stereo/mono chime track for demo MP4 streams and tests.
    pub fn synthesize_harmonic_chime(
        duration_ms: u32,
        sample_rate: u32,
        channels: u8,
        codec: AudioCodecProfile,
    ) -> Self {
        let sr = sample_rate.clamp(8_000, 96_000);
        let ch = channels.clamp(1, 2);
        let total_frames = ((duration_ms as u64 * sr as u64) / 1000) as usize;
        let mut samples = Vec::with_capacity(total_frames * ch as usize);
        for i in 0..total_frames {
            let t = (i as f32) / (sr as f32);
            let env = (1.0 - (i as f32 / (total_frames.max(1) as f32)) * 0.55).clamp(0.15, 1.0);
            let f1 = (t * 440.0 * core::f32::consts::TAU).sin();
            let f2 = (t * 659.25 * core::f32::consts::TAU).sin() * 0.5;
            let sample_l = ((f1 + f2) * 0.45 * env * 18_000.0).clamp(-32767.0, 32767.0) as i16;
            samples.push(sample_l);
            if ch == 2 {
                let f3 = (t * 554.37 * core::f32::consts::TAU).sin() * 0.6;
                let sample_r = ((f1 + f3) * 0.45 * env * 18_000.0).clamp(-32767.0, 32767.0) as i16;
                samples.push(sample_r);
            }
        }
        Self {
            sample_rate: sr,
            channels: ch,
            codec,
            samples,
        }
    }
}

impl AudioTrack {
    /// Extract the interleaved 16-bit PCM sample window for `[start_ms, start_ms + duration_ms)`.
    pub fn slice_pcm_window(&self, start_ms: u32, duration_ms: u32) -> Vec<i16> {
        if self.samples.is_empty() || self.sample_rate == 0 || self.channels == 0 {
            return Vec::new();
        }
        let ch = self.channels as usize;
        let sr = self.sample_rate as usize;
        let start_frame = ((start_ms as usize) * sr) / 1000;
        let span_frames = (((duration_ms.max(1) as usize) * sr) / 1000).max(1);
        let start_idx = (start_frame * ch).min(self.samples.len());
        let end_idx = ((start_frame + span_frames) * ch).min(self.samples.len());
        if start_idx >= end_idx {
            return Vec::new();
        }
        self.samples[start_idx..end_idx].to_vec()
    }

    /// Encode this track as a standard 44-byte header RIFF `WAVE` (`PCM16LE`) byte buffer.
    pub fn to_wav_bytes(&self) -> Vec<u8> {
        encode_pcm16_wav(self.sample_rate, self.channels, &self.samples)
    }
}

/// Encode interleaved 16-bit signed PCM samples into a RIFF `WAVE` (`fmt ` + `data`) byte buffer.
pub fn encode_pcm16_wav(sample_rate: u32, channels: u8, samples: &[i16]) -> Vec<u8> {
    let ch = u16::from(channels.clamp(1, 2));
    let sr = sample_rate.max(8_000);
    let data_bytes = (samples.len() * 2) as u32;
    let riff_size = 36u32.saturating_add(data_bytes);
    let byte_rate = sr.saturating_mul(u32::from(ch)).saturating_mul(2);
    let block_align = ch * 2;

    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM = 1
    out.extend_from_slice(&ch.to_le_bytes());
    out.extend_from_slice(&sr.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample = 16
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());
    for &s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Host operating-system audio output backend reported by [`AudioOutputBridge`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OsAudioBackend {
    /// Windows Multimedia `winmm.dll` (`waveOutGetNumDevs` / `sndPlaySoundA` / `WaveOut`).
    WinMmWaveOut,
    /// Linux ALSA PCM (`libasound` / `/proc/asound`).
    LinuxAlsaPcm,
    /// Software PCM mixer sink when no hardware audio output device is present.
    SoftwareMixerSink,
}

impl OsAudioBackend {
    /// Human-readable label for diagnostics and the media HUD.
    pub fn label(self) -> &'static str {
        match self {
            OsAudioBackend::WinMmWaveOut => "Windows WinMM WaveOut (PCM16)",
            OsAudioBackend::LinuxAlsaPcm => "Linux ALSA PCM (PCM16)",
            OsAudioBackend::SoftwareMixerSink => "Software PCM16 Mixer Sink",
        }
    }

    /// `true` when backed by a host OS audio subsystem (`WinMmWaveOut` or `LinuxAlsaPcm`).
    pub fn is_native_os_device(self) -> bool {
        !matches!(self, OsAudioBackend::SoftwareMixerSink)
    }
}

/// Probe the host operating system's native audio subsystem and return `(backend, device_count)`.
pub fn probe_os_audio_backend() -> (OsAudioBackend, u32) {
    #[cfg(target_os = "windows")]
    {
        use core::ffi::c_void;
        type HModule = *mut c_void;
        type FarProc = *const c_void;
        // SAFETY: these are the kernel32 signatures for LoadLibraryA and
        // GetProcAddress. They are only called with NUL-terminated names, and
        // both results are checked for null before use.
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn LoadLibraryA(lp_lib_file_name: *const u8) -> HModule;
            fn GetProcAddress(h_module: HModule, lp_proc_name: *const u8) -> FarProc;
        }
        // SAFETY: the DLL and function names are NUL-terminated literals, and
        // both handles are checked for null. winmm.dll is never freed, so the
        // function pointer stays valid, and waveOutGetNumDevs takes no arguments
        // and returns a u32, the type it is cast to.
        unsafe {
            let winmm = LoadLibraryA(b"winmm.dll\0".as_ptr());
            if !winmm.is_null() {
                let proc_ptr = GetProcAddress(winmm, b"waveOutGetNumDevs\0".as_ptr());
                if !proc_ptr.is_null() {
                    type WaveOutGetNumDevsFn = unsafe extern "system" fn() -> u32;
                    let get_num_devs: WaveOutGetNumDevsFn = core::mem::transmute(proc_ptr);
                    let count = get_num_devs();
                    return (OsAudioBackend::WinMmWaveOut, count);
                }
            }
        }
        (OsAudioBackend::WinMmWaveOut, 0)
    }
    #[cfg(target_os = "linux")]
    {
        let cards = std::fs::read_to_string("/proc/asound/cards").unwrap_or_default();
        let count = cards.lines().filter(|l| l.contains('[')).count() as u32;
        (OsAudioBackend::LinuxAlsaPcm, count)
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        (OsAudioBackend::SoftwareMixerSink, 0)
    }
}

/// Synchronized PCM audio output bridge for `<video>` playback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioOutputBridge {
    /// Active OS audio backend.
    pub backend: OsAudioBackend,
    /// Number of hardware audio output devices reported by the OS probe.
    pub hardware_devices_detected: u32,
    /// Whether audio output is muted.
    pub muted: bool,
    /// Playback volume percentage (`0..=100`).
    pub volume_percent: u8,
    /// Total audio windows submitted across playback steps.
    pub submitted_windows: u64,
    /// Total interleaved PCM samples mixed and submitted.
    pub submitted_samples: u64,
    /// Peak amplitude (`0..=100`) of the most recently submitted window.
    pub last_peak_level: u8,
}

impl Default for AudioOutputBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioOutputBridge {
    /// Probe the host audio subsystem and initialize an unmuted bridge at 100% volume.
    pub fn new() -> Self {
        let (backend, hardware_devices_detected) = probe_os_audio_backend();
        Self {
            backend,
            hardware_devices_detected,
            muted: false,
            volume_percent: 100,
            submitted_windows: 0,
            submitted_samples: 0,
            last_peak_level: 0,
        }
    }

    /// Slice `[pts_ms, pts_ms + window_ms)` from `track`, apply volume/mute scaling,
    /// update telemetry counters, and optionally dispatch an asynchronous in-memory RIFF WAVE
    /// buffer to `winmm.dll` (`sndPlaySoundA`) on Windows when `emit_to_os_device` is true.
    pub fn submit_track_window(
        &mut self,
        track: &AudioTrack,
        pts_ms: u32,
        window_ms: u32,
        emit_to_os_device: bool,
    ) -> usize {
        let mut pcm = track.slice_pcm_window(pts_ms, window_ms);
        if pcm.is_empty() {
            self.last_peak_level = 0;
            return 0;
        }
        let scale = if self.muted {
            0i32
        } else {
            i32::from(self.volume_percent.min(100))
        };
        let mut peak = 0i32;
        for sample in &mut pcm {
            let scaled = ((i32::from(*sample) * scale) / 100).clamp(-32768, 32767);
            *sample = scaled as i16;
            let amp = scaled.abs();
            if amp > peak {
                peak = amp;
            }
        }
        self.last_peak_level = ((peak * 100) / 32767).clamp(0, 100) as u8;
        self.submitted_windows = self.submitted_windows.saturating_add(1);
        self.submitted_samples = self.submitted_samples.saturating_add(pcm.len() as u64);

        #[cfg(target_os = "windows")]
        if emit_to_os_device && !self.muted && self.hardware_devices_detected > 0 {
            use core::ffi::c_void;
            type HModule = *mut c_void;
            type FarProc = *const c_void;
            // SAFETY: these are the kernel32 signatures for LoadLibraryA and
            // GetProcAddress. They are only called with NUL-terminated names, and
            // both results are checked for null before use.
            #[link(name = "kernel32")]
            unsafe extern "system" {
                fn LoadLibraryA(lp_lib_file_name: *const u8) -> HModule;
                fn GetProcAddress(h_module: HModule, lp_proc_name: *const u8) -> FarProc;
            }
            const SND_ASYNC: u32 = 0x0001;
            const SND_NODEFAULT: u32 = 0x0002;
            const SND_MEMORY: u32 = 0x0004;
            let wav = encode_pcm16_wav(track.sample_rate, track.channels, &pcm);
            // SAFETY: the DLL and function names are NUL-terminated literals, and
            // both handles are checked for null. winmm.dll is never freed, so the
            // function pointer stays valid, and sndPlaySoundA has the type it is cast to.
            unsafe {
                let winmm = LoadLibraryA(b"winmm.dll\0".as_ptr());
                if !winmm.is_null() {
                    let proc_ptr = GetProcAddress(winmm, b"sndPlaySoundA\0".as_ptr());
                    if !proc_ptr.is_null() {
                        type SndPlaySoundAFn =
                            unsafe extern "system" fn(lpsz_sound: *const u8, fu_sound: u32) -> i32;
                        let play_fn: SndPlaySoundAFn = core::mem::transmute(proc_ptr);
                        let _ = play_fn(wav.as_ptr(), SND_MEMORY | SND_ASYNC | SND_NODEFAULT);
                    }
                }
            }
        }
        let _ = emit_to_os_device;
        pcm.len()
    }
}

fn ima_adpcm_encode_step(sample: i16, predictor: &mut i32, step_index: &mut i32) -> u8 {
    let step = IMA_STEP_TABLE[*step_index as usize];
    let mut diff = (sample as i32) - *predictor;
    let mut nibble = 0u8;
    if diff < 0 {
        nibble = 8;
        diff = -diff;
    }
    let mut vpdiff = step >> 3;
    if diff >= step {
        nibble |= 4;
        diff -= step;
        vpdiff += step;
    }
    if diff >= (step >> 1) {
        nibble |= 2;
        diff -= step >> 1;
        vpdiff += step >> 1;
    }
    if diff >= (step >> 2) {
        nibble |= 1;
        vpdiff += step >> 2;
    }
    if (nibble & 8) != 0 {
        *predictor -= vpdiff;
    } else {
        *predictor += vpdiff;
    }
    *predictor = (*predictor).clamp(-32768, 32767);
    *step_index = (*step_index + IMA_INDEX_TABLE[nibble as usize]).clamp(0, 88);
    nibble & 0x0F
}

fn ima_adpcm_decode_step(nibble: u8, predictor: &mut i32, step_index: &mut i32) -> i16 {
    let step = IMA_STEP_TABLE[*step_index as usize];
    let mut vpdiff = step >> 3;
    if (nibble & 4) != 0 {
        vpdiff += step;
    }
    if (nibble & 2) != 0 {
        vpdiff += step >> 1;
    }
    if (nibble & 1) != 0 {
        vpdiff += step >> 2;
    }
    if (nibble & 8) != 0 {
        *predictor -= vpdiff;
    } else {
        *predictor += vpdiff;
    }
    *predictor = (*predictor).clamp(-32768, 32767);
    *step_index = (*step_index + IMA_INDEX_TABLE[(nibble & 0x0F) as usize]).clamp(0, 88);
    *predictor as i16
}

/// Encode an [`AudioTrack`] into a 16-byte `AAUD` header followed by compressed audio data.
pub fn encode_aaud_bitstream(track: &AudioTrack) -> Result<Vec<u8>, VideoError> {
    if track.channels == 0 || track.channels > 2 || track.sample_rate == 0 {
        return Err(VideoError::InvalidHeader);
    }
    let sample_count = track.samples.len();
    if sample_count > MAX_PACKET_BYTES / 2 {
        return Err(VideoError::PacketTooLarge {
            size: sample_count * 2,
        });
    }
    let (codec_byte, compressed) = match track.codec {
        AudioCodecProfile::Pcm16Le => {
            let mut raw = Vec::with_capacity(sample_count * 2);
            for &s in &track.samples {
                raw.extend_from_slice(&s.to_le_bytes());
            }
            (0u8, compress_zlib(&raw))
        }
        AudioCodecProfile::ImaAdpcm4 => {
            let mut raw = Vec::with_capacity((sample_count + 1) / 2);
            let ch = track.channels as usize;
            let mut pred = [0i32; 2];
            let mut idx = [0i32; 2];
            let mut packed_byte = 0u8;
            for (i, &s) in track.samples.iter().enumerate() {
                let c = i % ch;
                let nib = ima_adpcm_encode_step(s, &mut pred[c], &mut idx[c]);
                if (i & 1) == 0 {
                    packed_byte = nib;
                } else {
                    packed_byte |= nib << 4;
                    raw.push(packed_byte);
                }
            }
            if (sample_count & 1) != 0 {
                raw.push(packed_byte);
            }
            (1u8, compress_zlib(&raw))
        }
        AudioCodecProfile::FlacLossless => (2u8, encode_flac_stream(track)?),
        AudioCodecProfile::OpusMdct => (3u8, encode_mdct_audio_stream(track)?),
    };
    if compressed.len() > MAX_PACKET_BYTES {
        return Err(VideoError::PacketTooLarge {
            size: compressed.len(),
        });
    }
    let mut out = Vec::with_capacity(16 + compressed.len());
    out.extend_from_slice(b"AAUD");
    out.push(1u8);
    out.push(codec_byte);
    out.push(track.channels);
    out.push(0u8);
    out.extend_from_slice(&track.sample_rate.to_be_bytes());
    out.extend_from_slice(&(sample_count as u32).to_be_bytes());
    out.extend_from_slice(&compressed);
    Ok(out)
}

/// Decode an `AAUD`, `fLaC`, or `MDCT` audio bitstream back into an [`AudioTrack`].
pub fn decode_aaud_bitstream(data: &[u8]) -> Result<AudioTrack, VideoError> {
    if data.len() >= 4 && &data[0..4] == b"fLaC" {
        return decode_flac_stream(data);
    }
    if data.len() >= 4 && &data[0..4] == b"MDCT" {
        return decode_mdct_audio_stream(data);
    }
    if data.len() < 16 || &data[0..4] != b"AAUD" || data[4] != 1 {
        return Err(VideoError::InvalidHeader);
    }
    let codec = match data[5] {
        0 => AudioCodecProfile::Pcm16Le,
        1 => AudioCodecProfile::ImaAdpcm4,
        2 => return decode_flac_stream(&data[16..]),
        3 => return decode_mdct_audio_stream(&data[16..]),
        _ => return Err(VideoError::InvalidHeader),
    };
    let channels = data[6];
    if channels == 0 || channels > 2 {
        return Err(VideoError::InvalidHeader);
    }
    let sample_rate = u32::from_be_bytes([data[8], data[9], data[10], data[11]]);
    let sample_count = u32::from_be_bytes([data[12], data[13], data[14], data[15]]) as usize;
    if sample_rate == 0 || sample_count > MAX_PACKET_BYTES / 2 {
        return Err(VideoError::InvalidHeader);
    }
    let expected_raw = match codec {
        AudioCodecProfile::Pcm16Le => sample_count * 2,
        AudioCodecProfile::ImaAdpcm4 => (sample_count + 1) / 2,
        AudioCodecProfile::FlacLossless | AudioCodecProfile::OpusMdct => unreachable!(),
    };
    let raw = decompress_zlib_bounded(&data[16..], expected_raw.max(1))?;
    if raw.len() != expected_raw {
        return Err(VideoError::CorruptedBitstream);
    }
    let mut samples = Vec::with_capacity(sample_count);
    match codec {
        AudioCodecProfile::Pcm16Le => {
            for chunk in raw.chunks_exact(2) {
                samples.push(i16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
        AudioCodecProfile::ImaAdpcm4 => {
            let ch = channels as usize;
            let mut pred = [0i32; 2];
            let mut idx = [0i32; 2];
            for i in 0..sample_count {
                let byte = raw[i / 2];
                let nib = if (i & 1) == 0 {
                    byte & 0x0F
                } else {
                    (byte >> 4) & 0x0F
                };
                let c = i % ch;
                samples.push(ima_adpcm_decode_step(nib, &mut pred[c], &mut idx[c]));
            }
        }
        AudioCodecProfile::FlacLossless | AudioCodecProfile::OpusMdct => unreachable!(),
    }
    Ok(AudioTrack {
        sample_rate,
        channels,
        codec,
        samples,
    })
}

/// Pick a [`GpuVideoBackend`] from `hardware_enabled` and the compile target.
///
/// `false` returns [`GpuVideoBackend::SoftwareFallback`]. Otherwise Windows returns
/// [`GpuVideoBackend::D3d11Dxva2`], Linux returns [`GpuVideoBackend::LinuxVaApi`],
/// and any other target returns [`GpuVideoBackend::SoftwareFallback`].
/// No device is opened or enumerated.
pub fn probe_platform_gpu_backend(hardware_enabled: bool) -> GpuVideoBackend {
    if !hardware_enabled {
        return GpuVideoBackend::SoftwareFallback;
    }
    #[cfg(target_os = "windows")]
    {
        GpuVideoBackend::D3d11Dxva2
    }
    #[cfg(target_os = "linux")]
    {
        GpuVideoBackend::LinuxVaApi
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        GpuVideoBackend::SoftwareFallback
    }
}

/// Reject a width or height that this crate will not convert to NV12.
///
/// Zero is [`VideoError::InvalidHeader`]. A side above [`MAX_VIDEO_WIDTH`] or
/// [`MAX_VIDEO_HEIGHT`], or a product above [`MAX_VIDEO_PIXELS`], is
/// [`VideoError::DimensionsTooLarge`]. An odd side is [`VideoError::OddDimensions`].
pub fn validate_video_dimensions(width: u32, height: u32) -> Result<(), VideoError> {
    // Zero width or height is rejected before the pixel cap.
    if width == 0 || height == 0 {
        return Err(VideoError::InvalidHeader);
    }
    // Cap each side at 4096x2160 and the product at MAX_VIDEO_PIXELS.
    if width > MAX_VIDEO_WIDTH
        || height > MAX_VIDEO_HEIGHT
        || u64::from(width) * u64::from(height) > MAX_VIDEO_PIXELS
    {
        return Err(VideoError::DimensionsTooLarge { width, height });
    }
    // NV12 chroma is 4:2:0, so both sides must be even.
    if width % 2 != 0 || height % 2 != 0 {
        return Err(VideoError::OddDimensions { width, height });
    }
    Ok(())
}

/// One frame as packed `RGBA8888` plus a presentation time in milliseconds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoFrame {
    /// Frame width in pixels. Even, and at most [`MAX_VIDEO_WIDTH`].
    pub width: u32,
    /// Frame height in pixels. Even, and at most [`MAX_VIDEO_HEIGHT`].
    pub height: u32,
    /// Presentation time in milliseconds. Stored as given. Not computed from fps.
    pub pts_ms: u32,
    /// Caller flag. Encode also forces a full plane packet on index 0 and every 8th frame.
    pub keyframe: bool,
    /// `width * height * 4` bytes, in red, green, blue, alpha order.
    pub rgba: Vec<u8>,
}

impl VideoFrame {
    /// Check dimensions and require `rgba.len() == width * height * 4`.
    pub fn new(
        width: u32,
        height: u32,
        pts_ms: u32,
        keyframe: bool,
        rgba: Vec<u8>,
    ) -> Result<Self, VideoError> {
        validate_video_dimensions(width, height)?;
        let expected = (width as usize) * (height as usize) * 4;
        if rgba.len() != expected {
            return Err(VideoError::InvalidHeader);
        }
        Ok(Self {
            width,
            height,
            pts_ms,
            keyframe,
            rgba,
        })
    }

    /// Convert these RGBA bytes to NV12 by calling `Nv12Surface::from_rgba_bt709`.
    ///
    /// The Y and UV planes stay in the returned struct. This does not upload them to a GPU.
    /// `None` from that call becomes [`VideoError::InvalidHeader`].
    pub fn to_nv12_surface(&self) -> Result<Nv12Surface, VideoError> {
        Nv12Surface::from_rgba_bt709(self.width, self.height, &self.rgba)
            .ok_or(VideoError::InvalidHeader)
    }

    /// Build a frame from NV12 planes by calling `Nv12Surface::to_rgba_bt709`.
    ///
    /// Dimensions go through [`validate_video_dimensions`]. `pts_ms` and `keyframe` are stored
    /// as given. The conversion runs on the CPU. Alpha comes from that call.
    pub fn from_nv12_surface(
        surface: &Nv12Surface,
        pts_ms: u32,
        keyframe: bool,
    ) -> Result<Self, VideoError> {
        validate_video_dimensions(surface.width, surface.height)?;
        Ok(Self {
            width: surface.width,
            height: surface.height,
            pts_ms,
            keyframe,
            rgba: surface.to_rgba_bt709(),
        })
    }

    /// Blit and scale this frame into `dst_rgba` (`dst_width` by `dst_height`, row stride
    /// `dst_stride` bytes) at `(x, y, width, height)` through `anvil_gpu::Compositor`.
    #[allow(clippy::too_many_arguments)]
    pub fn blit_into(
        &self,
        dst_rgba: &mut [u8],
        dst_width: u32,
        dst_height: u32,
        dst_stride: usize,
        x: i32,
        y: i32,
        width: u32,
        height: u32,
        bilinear: bool,
    ) -> Result<(), VideoError> {
        let mut target = anvil_gpu::RgbaTarget::new(dst_rgba, dst_width, dst_height, dst_stride)
            .map_err(|_| VideoError::InvalidHeader)?;
        let src = anvil_gpu::VideoFrame::Rgba {
            width: self.width,
            height: self.height,
            stride: (self.width as usize) * 4,
            pixels: &self.rgba,
        };
        let desc = anvil_gpu::BlitDesc {
            x,
            y,
            width,
            height,
            filter: if bilinear {
                anvil_gpu::Filter::Bilinear
            } else {
                anvil_gpu::Filter::Nearest
            },
            clip: None,
        };
        anvil_gpu::Compositor::open()
            .blit_frame(&mut target, &src, &desc)
            .map_err(|_| VideoError::InvalidHeader)
    }
}

/// Present an [`Nv12Surface`] into packed RGBA8 through `anvil_gpu::Compositor`
/// (with `Filter::Nearest`), falling back to `present_nv12_on_device`.
pub fn present_nv12_via_compositor(surface: &Nv12Surface) -> Vec<u8> {
    let mut out = present_nv12_on_device(surface);
    if surface.width > 0 && surface.height > 0 {
        if let Ok(mut target) = anvil_gpu::RgbaTarget::new(
            &mut out,
            surface.width,
            surface.height,
            (surface.width as usize) * 4,
        ) {
            let src = anvil_gpu::VideoFrame::Nv12 {
                width: surface.width,
                height: surface.height,
                y_stride: surface.width as usize,
                y: &surface.y_plane,
                uv_stride: surface.width as usize,
                uv: &surface.uv_plane,
            };
            let desc = anvil_gpu::BlitDesc {
                x: 0,
                y: 0,
                width: surface.width,
                height: surface.height,
                filter: anvil_gpu::Filter::Nearest,
                clip: None,
            };
            let _ = anvil_gpu::Compositor::open().blit_frame(&mut target, &src, &desc);
        }
    }
    out
}

impl VideoFrame {
    #[cfg(test)]
    #[allow(dead_code)]
    fn _unused_frame_check(
        surface: &Nv12Surface,
        pts_ms: u32,
        keyframe: bool,
    ) -> Result<VideoFrame, VideoError> {
        Ok(VideoFrame {
            width: surface.width,
            height: surface.height,
            pts_ms,
            keyframe,
            rgba: surface.to_rgba_bt709(),
        })
    }
}

/// Frames decoded from one `AVID` bitstream, plus the header fields copied next to them.
///
/// `backend` is a label from [`probe_platform_gpu_backend`], not a measurement from a device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoClip {
    /// Width copied from the `AVID` header.
    pub width: u32,
    /// Height copied from the `AVID` header.
    pub height: u32,
    /// Frames per second copied from the `AVID` header, at least 1.
    pub fps: u32,
    /// Backend label stored by the decoder. Not a device handle.
    pub backend: GpuVideoBackend,
    /// Profile stored by the decoder. [`GpuVideoSession::decode_avid`] writes
    /// [`VideoCodecProfile::AnvilNv12Temporal`].
    pub profile: VideoCodecProfile,
    /// Frames in bitstream order.
    pub frames: Vec<VideoFrame>,
    /// Optional decoded audio track carried in a dual-track `AAVM` container.
    pub audio: Option<AudioTrack>,
}

impl VideoClip {
    /// `pts_ms` of the last frame, or 0 when `frames` is empty.
    ///
    /// This is not a sum of per-frame durations.
    pub fn duration_ms(&self) -> u32 {
        self.frames.last().map(|f| f.pts_ms).unwrap_or(0)
    }

    /// The first frame, if there is one.
    pub fn poster_frame(&self) -> Option<&VideoFrame> {
        self.frames.first()
    }

    /// How many frames have [`VideoFrame::keyframe`] set.
    ///
    /// After [`GpuVideoSession::decode_avid`], that flag is set on `I` packets only.
    pub fn keyframe_count(&self) -> usize {
        self.frames.iter().filter(|f| f.keyframe).count()
    }

    /// Latest frame whose `pts_ms` is less than or equal to `timestamp_ms`.
    ///
    /// An empty clip returns `None`. A timestamp earlier than every frame returns the
    /// first frame. A timestamp past the last frame returns the last frame. The search
    /// stops at the first frame with a greater `pts_ms`, so times must be non-decreasing.
    pub fn frame_at_ms(&self, timestamp_ms: u32) -> Option<&VideoFrame> {
        if self.frames.is_empty() {
            return None;
        }
        let mut chosen = &self.frames[0];
        for frame in &self.frames {
            if frame.pts_ms <= timestamp_ms {
                chosen = frame;
            } else {
                break;
            }
        }
        Some(chosen)
    }

    /// Peak audio amplitude (`0..=100`) at `timestamp_ms`, or `0` when no audio track is present.
    pub fn peak_audio_level_at_ms(&self, timestamp_ms: u32) -> u8 {
        self.audio
            .as_ref()
            .map(|a| a.peak_level_at_ms(timestamp_ms, 33))
            .unwrap_or(0)
    }
}

/// Fields copied from a container. `mdat` bytes are not decoded.
///
/// `brand` is file bytes 8..12, which are the `ftyp` major brand when the file starts
/// with an 8-byte box header. Width and height come from a `tkhd` payload of at least
/// 12 bytes (offsets 4 and 8). Timescale and duration come from an `mvhd` payload of
/// at least 12 bytes (offsets 4 and 8). Timescale is forced to at least 1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mp4Metadata {
    /// Four bytes at file offset 8.
    pub brand: [u8; 4],
    /// Width from `tkhd`, or 0 when no suitable `tkhd` was seen.
    pub width: u32,
    /// Height from `tkhd`, or 0 when no suitable `tkhd` was seen.
    pub height: u32,
    /// Timescale from `mvhd`. Starts at 1000 and is at least 1 after a parse.
    pub timescale: u32,
    /// Duration word from `mvhd`. This crate writes milliseconds there.
    pub duration_ms: u32,
    /// Payload length of the `mdat` box that the walk recorded.
    pub mdat_size: usize,
    /// Boxes counted by the walk, including nested ones.
    pub box_count: usize,
    /// `true` when a `tenc` payload of at least 24 bytes has a non-zero byte at offset 6.
    pub encrypted: bool,
    /// Four bytes at `schm` payload offset 4, when that payload is at least 8 bytes.
    pub protection_scheme: Option<[u8; 4]>,
    /// 16-byte key id at `tenc` payload offset 8, when that payload is at least 24 bytes.
    pub default_kid: Option<[u8; 16]>,
    /// 16 bytes at `pssh` payload offset 44, when that payload is at least 60 bytes.
    /// Used as the AES-128-CTR initial counter.
    pub iv: Option<[u8; 16]>,
    /// How many `pssh` boxes had a payload of at least 20 bytes.
    pub pssh_count: usize,
    /// `true` when an audio `hdlr` (`"soun"`) or `smhd` box was seen in `moov`.
    pub has_audio_track: bool,
    /// Audio sample rate from the `soun` handler or `AAUD` stream, when present.
    pub audio_sample_rate: u32,
    /// Audio channel count from the `soun` handler or `AAUD` stream, when present.
    pub audio_channels: u8,
}

/// Label stored beside the demo key list. Selecting a variant does not load a module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmeKeySystem {
    /// Label `org.w3.clearkey`. [`CdmSession::with_default_demo_key`] uses this.
    /// No key-system handshake is run.
    W3cClearKey,
    /// Label `org.anvil.mediafoundation.cdm`. No operating-system media API is called.
    OsMediaFoundationDrm,
    /// Label `org.anvil.external.cdm`. No plugin file is opened.
    ExternalPluginCdm,
}

impl EmeKeySystem {
    /// Stable identifier string for this label.
    pub fn identifier(self) -> &'static str {
        match self {
            EmeKeySystem::W3cClearKey => "org.w3.clearkey",
            EmeKeySystem::OsMediaFoundationDrm => "org.anvil.mediafoundation.cdm",
            EmeKeySystem::ExternalPluginCdm => "org.anvil.external.cdm",
        }
    }

    /// Display string for this label. The words in the string do not start a decoder or a plugin.
    pub fn label(self) -> &'static str {
        match self {
            EmeKeySystem::W3cClearKey => "W3C ClearKey (org.w3.clearkey + CENC AES-128-CTR)",
            EmeKeySystem::OsMediaFoundationDrm => "OS Media Foundation / DRM Bridge",
            EmeKeySystem::ExternalPluginCdm => "External CDM Host Plugin Slot",
        }
    }
}

/// In-memory list of 16-byte key ids and 16-byte AES-128 keys.
///
/// This is not a CDM and it does not talk to a license server. Decode looks a key up
/// here and passes it to [`cenc_decrypt_sample`], which calls `aes128_ctr`.
/// The IV is not stored in this list. It comes from `pssh` at decrypt time, or is 16 zero bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CdmSession {
    /// Label only. See [`EmeKeySystem`].
    key_system: EmeKeySystem,
    /// Pairs of (key id, key). Both halves are 16 bytes. At most 128 entries.
    keys: Vec<([u8; 16], [u8; 16])>,
    /// How many encrypted containers [`GpuVideoSession::decode_video_auto`] has decrypted.
    /// Not a count of frames.
    decrypted_samples: u64,
}

impl Default for CdmSession {
    /// Same as [`CdmSession::with_default_demo_key`].
    fn default() -> Self {
        Self::with_default_demo_key()
    }
}

impl CdmSession {
    /// Empty key list, zero decrypt count, and the given label.
    pub fn new(key_system: EmeKeySystem) -> Self {
        Self {
            key_system,
            keys: Vec::new(),
            decrypted_samples: 0,
        }
    }

    /// Session labeled [`EmeKeySystem::W3cClearKey`] with one demo entry.
    ///
    /// The entry is [`DEFAULT_EME_DEMO_KID`] and [`DEFAULT_EME_DEMO_KEY`].
    pub fn with_default_demo_key() -> Self {
        let mut session = Self::new(EmeKeySystem::W3cClearKey);
        // Fixed demo key id and key compiled into this file. Not read from disk.
        session.provide_license_key(DEFAULT_EME_DEMO_KID, DEFAULT_EME_DEMO_KEY);
        session
    }

    /// Label stored on this session. It does not select a real key system.
    pub fn key_system(&self) -> EmeKeySystem {
        self.key_system
    }

    /// Replace the label. Keys already stored are left as they are.
    pub fn set_key_system(&mut self, key_system: EmeKeySystem) {
        self.key_system = key_system;
    }

    /// Store a 16-byte AES-128 key for a 16-byte key id.
    ///
    /// An id that is already present is overwritten. A new id is appended.
    /// The key is kept in this process only.
    pub fn provide_license_key(&mut self, key_id: [u8; 16], key: [u8; 16]) {
        if let Some(existing) = self.keys.iter_mut().find(|(kid, _)| *kid == key_id) {
            existing.1 = key;
            return;
        }
        // At most 128 keys. Drop the oldest entry before pushing past that cap.
        if self.keys.len() >= 128 {
            self.keys.remove(0);
        }
        self.keys.push((key_id, key));
    }

    /// Remove every entry with this key id. Returns whether the list shrank.
    pub fn remove_key(&mut self, key_id: &[u8; 16]) -> bool {
        let before = self.keys.len();
        self.keys.retain(|(kid, _)| kid != key_id);
        self.keys.len() != before
    }

    /// Drop every stored key. The label and the decrypt counter stay.
    pub fn clear_keys(&mut self) {
        self.keys.clear();
    }

    /// Whether `key_id` is present.
    pub fn has_key(&self, key_id: &[u8; 16]) -> bool {
        self.keys.iter().any(|(kid, _)| kid == key_id)
    }

    /// Copy of the 16-byte key for `key_id`, if it is present.
    pub fn lookup_key(&self, key_id: &[u8; 16]) -> Option<[u8; 16]> {
        self.keys
            .iter()
            .find(|(kid, _)| kid == key_id)
            .map(|(_, k)| *k)
    }

    /// How many key entries are stored, at most 128.
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    /// How many encrypted containers [`GpuVideoSession::decode_video_auto`] decrypted.
    ///
    /// Not a frame count. A missing key does not increment it.
    pub fn decrypted_samples(&self) -> u64 {
        self.decrypted_samples
    }
}

/// CPU encode and decode state for `AVID` and for the small MP4 wrapper around it.
///
/// The name says GPU. What it stores is a [`GpuVideoBackend`] label, two boolean flags,
/// a count of frames passed through encode, a count of frames passed through decode,
/// and a [`CdmSession`] that starts with the fixed demo key. Encode and decode convert
/// NV12 in memory with `anvil-gpu` and compress with `anvil-compress`. They do not open
/// a device, and the flags do not change the plane math. Either flag updates the stored
/// backend label when the session is created or when [`GpuVideoSession::set_hardware_acceleration`]
/// runs. The encode flag also writes the `AVID` flag byte. The decode flag is what
/// [`GpuVideoSession::decode_avid`] passes to [`probe_platform_gpu_backend`] for the clip.
#[derive(Clone, Debug)]
pub struct GpuVideoSession {
    /// Label from [`probe_platform_gpu_backend`]. Not a device handle.
    backend: GpuVideoBackend,
    /// Stored flag. Decode still runs on the CPU.
    hardware_decode_enabled: bool,
    /// Stored flag. Encode still runs on the CPU. Also selects the `AVID` flag byte.
    hardware_encode_enabled: bool,
    /// Frames emitted by [`Self::decode_avid`].
    frames_decoded: u64,
    /// Frames accepted by [`Self::encode_avid`].
    frames_encoded: u64,
    /// Demo key list. Starts as [`CdmSession::with_default_demo_key`].
    cdm_session: CdmSession,
}

impl Default for GpuVideoSession {
    /// [`GpuVideoSession::new`] with both flags set to `true`.
    fn default() -> Self {
        Self::new(true, true)
    }
}

impl GpuVideoSession {
    /// Store the flags, set the backend label from their logical or, zero both counters,
    /// and install the compiled demo key.
    ///
    /// Neither flag creates a device. The demo key is [`DEFAULT_EME_DEMO_KEY`], a fixed
    /// array, not bytes read from disk.
    pub fn new(hardware_decode: bool, hardware_encode: bool) -> Self {
        let backend = probe_platform_gpu_backend(hardware_decode || hardware_encode);
        Self {
            backend,
            hardware_decode_enabled: hardware_decode,
            hardware_encode_enabled: hardware_encode,
            frames_decoded: 0,
            frames_encoded: 0,
            cdm_session: CdmSession::with_default_demo_key(),
        }
    }

    /// Borrow the in-memory key list.
    pub fn cdm_session(&self) -> &CdmSession {
        &self.cdm_session
    }

    /// Borrow the in-memory key list for updates.
    pub fn cdm_session_mut(&mut self) -> &mut CdmSession {
        &mut self.cdm_session
    }

    /// Backend label currently stored.
    pub fn backend(&self) -> GpuVideoBackend {
        self.backend
    }

    /// The stored decode flag. `true` does not mean a GPU decoded anything.
    pub fn is_hardware_decode_enabled(&self) -> bool {
        self.hardware_decode_enabled
    }

    /// The stored encode flag. `true` does not mean a GPU encoded anything.
    pub fn is_hardware_encode_enabled(&self) -> bool {
        self.hardware_encode_enabled
    }

    /// Replace both flags and recompute the backend label with [`probe_platform_gpu_backend`].
    ///
    /// Counters and keys are left as they are. No device is opened.
    pub fn set_hardware_acceleration(&mut self, decode: bool, encode: bool) {
        self.hardware_decode_enabled = decode;
        self.hardware_encode_enabled = encode;
        self.backend = probe_platform_gpu_backend(decode || encode);
    }

    /// Frames emitted by [`Self::decode_avid`] on this session.
    ///
    /// The count advances as each frame is pushed. A later error in that call does not
    /// subtract frames already counted.
    pub fn frames_decoded(&self) -> u64 {
        self.frames_decoded
    }

    /// Frames appended by [`Self::encode_avid`] on this session.
    ///
    /// The count advances as each frame is written. A later error in that call does not
    /// subtract frames already counted.
    pub fn frames_encoded(&self) -> u64 {
        self.frames_encoded
    }

    /// Write an `AVID` bitstream for `frames`.
    ///
    /// The header is big-endian: magic `AVID`, version `u16` 1, one flag byte, one zero
    /// byte, width, height, `fps` (stored as at least 1), and the frame count. The flag
    /// byte is 1 only when the encode flag is set and [`GpuVideoBackend::is_hardware_accelerated`]
    /// is true for the stored label. Decode does not read that byte. Planes are still built
    /// on the CPU.
    ///
    /// A frame is marked `I` when its index is 0, [`VideoFrame::keyframe`] is set, or the
    /// index is a multiple of 8. An `I` packet, or a packet with no previous planes, is the
    /// Y plane followed by the UV plane from `Nv12Surface::from_rgba_bt709`. Any other frame
    /// is `P`: each plane byte minus the previous plane byte with `wrapping_sub`. There is
    /// no motion search. `compress_zlib` from `anvil-compress` wraps that buffer. The record
    /// is the type byte, `pts_ms`, the compressed length, and the compressed bytes.
    /// Every frame must match the first frame's dimensions. An empty slice is
    /// [`VideoError::InvalidHeader`]. `frames_encoded` grows by one per written frame.
    pub fn encode_avid(&mut self, fps: u32, frames: &[VideoFrame]) -> Result<Vec<u8>, VideoError> {
        if frames.is_empty() {
            return Err(VideoError::InvalidHeader);
        }
        // Reject a clip longer than MAX_VIDEO_FRAMES.
        if frames.len() > MAX_VIDEO_FRAMES as usize {
            return Err(VideoError::FrameCountTooLarge {
                count: frames.len() as u32,
            });
        }
        let width = frames[0].width;
        let height = frames[0].height;
        validate_video_dimensions(width, height)?;

        let mut out = Vec::new();
        out.extend_from_slice(b"AVID");
        out.extend_from_slice(&1u16.to_be_bytes()); // version 1
        let hw_flag = if self.hardware_encode_enabled && self.backend.is_hardware_accelerated() {
            1u8
        } else {
            0u8
        };
        out.push(hw_flag);
        out.push(0u8); // reserved
        out.extend_from_slice(&width.to_be_bytes());
        out.extend_from_slice(&height.to_be_bytes());
        out.extend_from_slice(&fps.max(1).to_be_bytes());
        out.extend_from_slice(&(frames.len() as u32).to_be_bytes());

        let mut prev_nv12: Option<Nv12Surface> = None;

        for (idx, frame) in frames.iter().enumerate() {
            if frame.width != width || frame.height != height {
                return Err(VideoError::InvalidHeader);
            }
            let curr_nv12 = if self.hardware_encode_enabled {
                convert_rgba_on_device(frame.width, frame.height, &frame.rgba)
                    .ok_or(VideoError::InvalidHeader)?
            } else {
                frame.to_nv12_surface()?
            };
            let is_keyframe = idx == 0 || frame.keyframe || (idx % 8 == 0);

            let mut raw_plane_payload =
                Vec::with_capacity(curr_nv12.y_plane.len() + curr_nv12.uv_plane.len());

            if is_keyframe || prev_nv12.is_none() {
                raw_plane_payload.extend_from_slice(&curr_nv12.y_plane);
                raw_plane_payload.extend_from_slice(&curr_nv12.uv_plane);
            } else if let Some(ref prev) = prev_nv12 {
                for (&c, &p) in curr_nv12.y_plane.iter().zip(prev.y_plane.iter()) {
                    raw_plane_payload.push(c.wrapping_sub(p));
                }
                for (&c, &p) in curr_nv12.uv_plane.iter().zip(prev.uv_plane.iter()) {
                    raw_plane_payload.push(c.wrapping_sub(p));
                }
            }

            let compressed = compress_zlib(&raw_plane_payload);
            // Reject a zlib packet larger than MAX_PACKET_BYTES (16 MiB).
            if compressed.len() > MAX_PACKET_BYTES {
                return Err(VideoError::PacketTooLarge {
                    size: compressed.len(),
                });
            }

            out.push(if is_keyframe { b'I' } else { b'P' });
            out.extend_from_slice(&frame.pts_ms.to_be_bytes());
            out.extend_from_slice(&(compressed.len() as u32).to_be_bytes());
            out.extend_from_slice(&compressed);

            prev_nv12 = Some(curr_nv12);
            self.frames_encoded += 1;
        }

        Ok(out)
    }

    /// Read an `AVID` bitstream into a [`VideoClip`].
    ///
    /// Requires magic `AVID` and version 1. The flag byte at offset 6 is not read.
    /// An `I` packet replaces the reference Y and UV planes. A `P` packet adds each
    /// residual byte with `wrapping_add`. The first frame must be `I`. Packet bytes
    /// are inflated with `decompress_zlib_bounded`, capped at the plane size, then
    /// turned back into RGBA with [`VideoFrame::from_nv12_surface`]. The clip profile
    /// is [`VideoCodecProfile::AnvilNv12Temporal`]. `backend` is
    /// [`probe_platform_gpu_backend`] of the stored decode flag, not a device result.
    /// `frames_decoded` grows by one per emitted frame.
    pub fn decode_avid(&mut self, data: &[u8]) -> Result<VideoClip, VideoError> {
        let (data, audio_slice) = if data.len() >= 12 && &data[0..4] == b"AAVM" {
            let avid_len = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
            let aaud_len = u32::from_be_bytes([data[8], data[9], data[10], data[11]]) as usize;
            if avid_len > MAX_PACKET_BYTES
                || aaud_len > MAX_PACKET_BYTES
                || 12 + avid_len + aaud_len > data.len()
            {
                return Err(VideoError::CorruptedBitstream);
            }
            let aaud = if aaud_len > 0 {
                Some(&data[12 + avid_len..12 + avid_len + aaud_len])
            } else {
                None
            };
            (&data[12..12 + avid_len], aaud)
        } else {
            (data, None)
        };
        // Magic through frame count is 24 bytes. Offset 6 (flag) and 7 (reserved) are unread.
        if data.len() < 24 || &data[0..4] != b"AVID" {
            return Err(VideoError::InvalidHeader);
        }
        let version = u16::from_be_bytes([data[4], data[5]]);
        if version != 1 {
            return Err(VideoError::InvalidHeader);
        }
        let width = u32::from_be_bytes([data[8], data[9], data[10], data[11]]);
        let height = u32::from_be_bytes([data[12], data[13], data[14], data[15]]);
        let fps = u32::from_be_bytes([data[16], data[17], data[18], data[19]]).max(1);
        let frame_count = u32::from_be_bytes([data[20], data[21], data[22], data[23]]);

        validate_video_dimensions(width, height)?;
        // Reject a declared count of zero or a count above MAX_VIDEO_FRAMES.
        if frame_count == 0 || frame_count > MAX_VIDEO_FRAMES {
            return Err(VideoError::FrameCountTooLarge { count: frame_count });
        }

        let y_len = (width as usize) * (height as usize);
        let uv_len = ((width / 2) as usize) * ((height / 2) as usize) * 2;
        let plane_total = y_len + uv_len;

        let mut cursor = 24usize;
        let mut ref_surface = Nv12Surface::new(width, height);
        let mut frames = Vec::with_capacity(frame_count as usize);

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

            // Reject a packet above MAX_PACKET_BYTES or one that runs past the buffer.
            if pkt_len > MAX_PACKET_BYTES || cursor + pkt_len > data.len() {
                return Err(VideoError::PacketTooLarge { size: pkt_len });
            }

            let raw_planes = decompress_zlib_bounded(&data[cursor..cursor + pkt_len], plane_total)?;
            cursor += pkt_len;

            if raw_planes.len() != plane_total {
                return Err(VideoError::CorruptedBitstream);
            }

            let is_keyframe = match frame_type {
                b'I' => true,
                b'P' if frame_idx > 0 => false,
                _ => return Err(VideoError::CorruptedBitstream),
            };

            if is_keyframe {
                ref_surface.y_plane.copy_from_slice(&raw_planes[..y_len]);
                ref_surface.uv_plane.copy_from_slice(&raw_planes[y_len..]);
            } else {
                for (dst, &delta) in ref_surface.y_plane.iter_mut().zip(&raw_planes[..y_len]) {
                    *dst = dst.wrapping_add(delta);
                }
                for (dst, &delta) in ref_surface.uv_plane.iter_mut().zip(&raw_planes[y_len..]) {
                    *dst = dst.wrapping_add(delta);
                }
            }

            let decoded_frame = if self.hardware_decode_enabled {
                let rgba = present_nv12_via_compositor(&ref_surface);
                VideoFrame::new(
                    ref_surface.width,
                    ref_surface.height,
                    pts_ms,
                    is_keyframe,
                    rgba,
                )?
            } else {
                VideoFrame::from_nv12_surface(&ref_surface, pts_ms, is_keyframe)?
            };
            frames.push(decoded_frame);
            self.frames_decoded += 1;
        }

        let active_backend = probe_platform_gpu_backend(self.hardware_decode_enabled);
        Ok(VideoClip {
            width,
            height,
            fps,
            backend: active_backend,
            profile: VideoCodecProfile::AnvilNv12Temporal,
            frames,
            audio: match audio_slice {
                Some(raw_aaud) => Some(decode_aaud_bitstream(raw_aaud)?),
                None => None,
            },
        })
    }

    /// Encode `frames` as `AVID`, then wrap that bitstream with [`wrap_mp4_container`].
    ///
    /// Duration written into `mvhd` is the last frame's `pts_ms`, or 0 when that is missing.
    /// [`Self::encode_avid`] already rejects an empty list, so the first frame exists here.
    pub fn encode_mp4(&mut self, fps: u32, frames: &[VideoFrame]) -> Result<Vec<u8>, VideoError> {
        let avid_stream = self.encode_avid(fps, frames)?;
        let width = frames[0].width;
        let height = frames[0].height;
        let duration_ms = frames.last().map(|f| f.pts_ms).unwrap_or(0);
        Ok(wrap_mp4_container(width, height, duration_ms, &avid_stream))
    }

    /// Encode `frames` as `AVID` and an optional [`AudioTrack`] as `AAUD`, then mux both into an MP4.
    pub fn encode_av_mp4(
        &mut self,
        fps: u32,
        frames: &[VideoFrame],
        audio: Option<&AudioTrack>,
    ) -> Result<Vec<u8>, VideoError> {
        let avid_stream = self.encode_avid(fps, frames)?;
        let width = frames[0].width;
        let height = frames[0].height;
        let duration_ms = frames.last().map(|f| f.pts_ms).unwrap_or(0);
        let aaud_stream = match audio {
            Some(track) => Some(encode_aaud_bitstream(track)?),
            None => None,
        };
        Ok(wrap_av_mp4_container(
            width,
            height,
            duration_ms,
            &avid_stream,
            aaud_stream.as_deref(),
        ))
    }

    /// Encode `frames` as `AVID`, then wrap that bitstream as one AES-128-CTR sample.
    ///
    /// `key` is 16 bytes. `iv` is the 16-byte initial counter. `key_id` is the 16-byte id
    /// written into `tenc` and `pssh`. Encrypt goes through [`wrap_encrypted_mp4_container`],
    /// which calls [`cenc_encrypt_sample`] and therefore `aes128_ctr`. The same call decrypts.
    /// This does not write per-sample IV boxes.
    pub fn encode_encrypted_mp4(
        &mut self,
        fps: u32,
        frames: &[VideoFrame],
        key_id: [u8; 16],
        key: [u8; 16],
        iv: [u8; 16],
    ) -> Result<Vec<u8>, VideoError> {
        let avid_stream = self.encode_avid(fps, frames)?;
        let width = frames[0].width;
        let height = frames[0].height;
        let duration_ms = frames.last().map(|f| f.pts_ms).unwrap_or(0);
        Ok(wrap_encrypted_mp4_container(
            width,
            height,
            duration_ms,
            key_id,
            key,
            iv,
            &avid_stream,
        ))
    }

    fn decode_payload_auto(&mut self, payload: &[u8]) -> Result<VideoClip, VideoError> {
        let hw = self.hardware_decode_enabled;
        if payload.len() >= 5 && &payload[0..5] == b"\x00\x00\x00\x01\x67" {
            let mut clip = decode_h264_annexb(payload, hw)?;
            self.frames_decoded += clip.frames.len() as u64;
            clip.backend = probe_platform_gpu_backend(hw);
            return Ok(clip);
        }
        if payload.len() >= 3 && &payload[0..3] == b"\x12\x00\x0A" {
            let mut clip = decode_av1_obu_stream(payload, hw)?;
            self.frames_decoded += clip.frames.len() as u64;
            clip.backend = probe_platform_gpu_backend(hw);
            return Ok(clip);
        }
        if payload.len() >= 6 && (payload[0] & 1) == 0 && &payload[3..6] == &[0x9D, 0x01, 0x2A] {
            let mut clip = decode_vp8_rfc6386_stream(payload, hw)?;
            self.frames_decoded += clip.frames.len() as u64;
            clip.backend = probe_platform_gpu_backend(hw);
            return Ok(clip);
        }
        if payload.len() >= 4 && &payload[0..4] == &[0x82, 0x49, 0x83, 0x42] {
            let mut clip = decode_vp9_stream(payload, hw)?;
            self.frames_decoded += clip.frames.len() as u64;
            clip.backend = probe_platform_gpu_backend(hw);
            return Ok(clip);
        }
        if payload.len() >= 4 && &payload[0..4] == b"ADCT" {
            let mut clip = decode_dct_mc_bitstream(payload, hw)?;
            self.frames_decoded += clip.frames.len() as u64;
            clip.backend = probe_platform_gpu_backend(hw);
            return Ok(clip);
        }
        self.decode_avid(payload)
    }

    /// Decode a buffer that is an MP4 (`ftyp`), fragmented MP4 (`moof`), WebM (`\x1A\x45\xDF\xA3`),
    /// H.264 Annex-B (`\x00\x00\x00\x01\x67`), AV1/VP9 OBU (`\x12\x00\x0A`), `ADCT`, or `AVID` bitstream.
    pub fn decode_video_auto(&mut self, data: &[u8]) -> Result<VideoClip, VideoError> {
        if data.len() >= 4 && data[0..4] == [0x1A, 0x45, 0xDF, 0xA3] {
            let (_meta, video_payload) = parse_webm_container(data)?;
            return self.decode_payload_auto(video_payload);
        }
        if data.len() >= 8 && &data[4..8] == b"moof" {
            let seg = inspect_fmp4_segment(data)?;
            if !seg.has_media_fragment {
                return Err(VideoError::InvalidHeader);
            }
            let mdat_payload =
                extract_top_level_box_payload(data, b"mdat").ok_or(VideoError::InvalidHeader)?;
            return self.decode_payload_auto(mdat_payload);
        }
        if data.len() >= 8 && &data[4..8] == b"ftyp" {
            let meta = inspect_mp4_metadata(data)?;
            let payload = parse_mp4_container(data)?;
            if meta.encrypted {
                // No key id in the file falls back to the compiled demo id, not a key file.
                let kid = meta.default_kid.unwrap_or(DEFAULT_EME_DEMO_KID);
                let Some(key) = self.cdm_session.lookup_key(&kid) else {
                    return Err(VideoError::MissingDecryptionKey { kid });
                };
                let iv = meta.iv.unwrap_or([0u8; 16]);
                let decrypted = cenc_decrypt_sample(&key, &iv, payload);
                self.cdm_session.decrypted_samples =
                    self.cdm_session.decrypted_samples.saturating_add(1);
                return self.decode_payload_auto(&decrypted);
            }
            return self.decode_payload_auto(payload);
        }
        self.decode_payload_auto(data)
    }
}

/// Encrypt one sample with AES-128-CTR.
///
/// `key` is 16 bytes. `iv` is the 16-byte initial counter. This calls `aes128_ctr`.
/// [`cenc_decrypt_sample`] calls that same function. No padding is added.
pub fn cenc_encrypt_sample(key: &[u8; 16], iv: &[u8; 16], plaintext: &[u8]) -> Vec<u8> {
    aes128_ctr(key, iv, plaintext)
}

/// Decrypt one sample with AES-128-CTR.
///
/// `key` is 16 bytes. `iv` is the 16-byte initial counter. This calls `aes128_ctr`,
/// the same function [`cenc_encrypt_sample`] calls. No padding is removed. A wrong key
/// still returns a buffer of the same length.
pub fn cenc_decrypt_sample(key: &[u8; 16], iv: &[u8; 16], ciphertext: &[u8]) -> Vec<u8> {
    aes128_ctr(key, iv, ciphertext)
}

/// Put `mdat_payload` in an MP4 with `ftyp`, `moov` (`mvhd` and `trak`/`tkhd`), and `mdat`.
///
/// Each box is written by [`write_mp4_box`]: a big-endian `u32` size that includes the
/// 8-byte header, then the four-byte type, then the payload. `ftyp` is major brand `isom`,
/// minor version 512, and compatible brands `isom` and `mp42` (20 payload bytes).
/// `mvhd` is 12 bytes: flags word 0, timescale 1000, then `duration_ms`.
/// `tkhd` is 12 bytes: flags word 1, then `width` and `height` as `u32`.
/// `mdat` is `mdat_payload` unchanged. No `mdhd`, `hdlr`, or `stbl` is written.
pub fn wrap_mp4_container(
    width: u32,
    height: u32,
    duration_ms: u32,
    mdat_payload: &[u8],
) -> Vec<u8> {
    let mut out = Vec::new();

    // `ftyp` payload is 20 bytes: major brand, minor version, two compatible brands.
    let mut ftyp = Vec::new();
    ftyp.extend_from_slice(b"isom");
    ftyp.extend_from_slice(&512u32.to_be_bytes());
    ftyp.extend_from_slice(b"isommp42");
    write_mp4_box(&mut out, *b"ftyp", &ftyp);

    // `moov` contains `mvhd` and `trak`. `trak` contains only `tkhd`.
    let mut mvhd = Vec::new();
    mvhd.extend_from_slice(&0u32.to_be_bytes()); // version + flags
    mvhd.extend_from_slice(&1000u32.to_be_bytes()); // timescale = 1000 ms
    mvhd.extend_from_slice(&duration_ms.to_be_bytes());

    let mut tkhd = Vec::new();
    tkhd.extend_from_slice(&1u32.to_be_bytes()); // flags word 1, not a full ISO tkhd
    tkhd.extend_from_slice(&width.to_be_bytes());
    tkhd.extend_from_slice(&height.to_be_bytes());

    let mut trak = Vec::new();
    write_mp4_box(&mut trak, *b"tkhd", &tkhd);

    let mut moov = Vec::new();
    write_mp4_box(&mut moov, *b"mvhd", &mvhd);
    write_mp4_box(&mut moov, *b"trak", &trak);
    write_mp4_box(&mut out, *b"moov", &moov);

    // `mdat` payload is the plaintext bitstream. This function does not encrypt it.
    write_mp4_box(&mut out, *b"mdat", mdat_payload);
    out
}

/// Mux an `AVID` video bitstream and an optional `AAUD` audio bitstream into an ISOBMFF MP4 container.
///
/// When `aaud_payload` is `Some`, a second `trak` box (`mdia` -> `hdlr` `"soun"` + `minf` -> `smhd`)
/// is written into `moov` and `mdat` carries a dual-chunk `AAVM` payload (`b"AAVM"`, `avid_len`, `aaud_len`).
pub fn wrap_av_mp4_container(
    width: u32,
    height: u32,
    duration_ms: u32,
    avid_payload: &[u8],
    aaud_payload: Option<&[u8]>,
) -> Vec<u8> {
    let Some(aaud) = aaud_payload else {
        return wrap_mp4_container(width, height, duration_ms, avid_payload);
    };

    let mut out = Vec::new();
    let mut ftyp = Vec::new();
    ftyp.extend_from_slice(b"isom");
    ftyp.extend_from_slice(&512u32.to_be_bytes());
    ftyp.extend_from_slice(b"isommp42");
    write_mp4_box(&mut out, *b"ftyp", &ftyp);

    let mut mvhd = Vec::new();
    mvhd.extend_from_slice(&0u32.to_be_bytes());
    mvhd.extend_from_slice(&1000u32.to_be_bytes());
    mvhd.extend_from_slice(&duration_ms.to_be_bytes());

    let mut tkhd = Vec::new();
    tkhd.extend_from_slice(&1u32.to_be_bytes());
    tkhd.extend_from_slice(&width.to_be_bytes());
    tkhd.extend_from_slice(&height.to_be_bytes());

    let mut video_trak = Vec::new();
    write_mp4_box(&mut video_trak, *b"tkhd", &tkhd);

    let (sample_rate, channels) = if aaud.len() >= 16 && &aaud[0..4] == b"AAUD" {
        (
            u32::from_be_bytes([aaud[8], aaud[9], aaud[10], aaud[11]]),
            aaud[6],
        )
    } else {
        (44_100, 2)
    };

    let mut hdlr = Vec::new();
    hdlr.extend_from_slice(&0u32.to_be_bytes());
    hdlr.extend_from_slice(b"soun");
    hdlr.extend_from_slice(&sample_rate.to_be_bytes());
    hdlr.push(channels);
    hdlr.push(0);

    let smhd = [0u8; 8];
    let mut minf = Vec::new();
    write_mp4_box(&mut minf, *b"smhd", &smhd);

    let mut mdia = Vec::new();
    write_mp4_box(&mut mdia, *b"hdlr", &hdlr);
    write_mp4_box(&mut mdia, *b"minf", &minf);

    let mut audio_trak = Vec::new();
    write_mp4_box(&mut audio_trak, *b"mdia", &mdia);

    let mut moov = Vec::new();
    write_mp4_box(&mut moov, *b"mvhd", &mvhd);
    write_mp4_box(&mut moov, *b"trak", &video_trak);
    write_mp4_box(&mut moov, *b"trak", &audio_trak);
    write_mp4_box(&mut out, *b"moov", &moov);

    let mut mdat_payload = Vec::with_capacity(12 + avid_payload.len() + aaud.len());
    mdat_payload.extend_from_slice(b"AAVM");
    mdat_payload.extend_from_slice(&(avid_payload.len() as u32).to_be_bytes());
    mdat_payload.extend_from_slice(&(aaud.len() as u32).to_be_bytes());
    mdat_payload.extend_from_slice(avid_payload);
    mdat_payload.extend_from_slice(aaud);
    write_mp4_box(&mut out, *b"mdat", &mdat_payload);
    out
}

/// Put one AES-128-CTR sample in an MP4 and record `cenc` labels in `moov`.
///
/// Boxes match [`wrap_mp4_container`], with two changes. Compatible brands are `isom`
/// and `cenc`. `trak` also contains `sinf`, which contains `schm` and `tenc`. `moov`
/// also contains `pssh`. `schm` is flags word 0, the four bytes `cenc`, and scheme
/// version `0x00010000`. `tenc` is flags word 0, two zero bytes, a protected byte 1,
/// an IV-size byte 16, and the 16-byte `key_id`. `pssh` is version byte 1, flags 0,
/// [`W3C_CLEARKEY_SYSTEM_ID`], a key-id count of 1, `key_id`, the IV length, and `iv`.
///
/// `key` is 16 bytes. `iv` is the 16-byte CTR counter. The `mdat` body is
/// [`cenc_encrypt_sample`] of `mdat_plaintext`, so encrypt and decrypt share `aes128_ctr`.
/// The whole bitstream is one sample. No `senc`, `saiz`, or `saio` box is written.
pub fn wrap_encrypted_mp4_container(
    width: u32,
    height: u32,
    duration_ms: u32,
    key_id: [u8; 16],
    key: [u8; 16],
    iv: [u8; 16],
    mdat_plaintext: &[u8],
) -> Vec<u8> {
    let mut out = Vec::new();

    // `ftyp` payload: major brand `isom`, minor version 512, brands `isom` and `cenc`.
    let mut ftyp = Vec::new();
    ftyp.extend_from_slice(b"isom");
    ftyp.extend_from_slice(&512u32.to_be_bytes());
    ftyp.extend_from_slice(b"isomcenc");
    write_mp4_box(&mut out, *b"ftyp", &ftyp);

    // Short `mvhd`: flags, timescale 1000, duration. Same 12-byte layout as the plain muxer.
    let mut mvhd = Vec::new();
    mvhd.extend_from_slice(&0u32.to_be_bytes());
    mvhd.extend_from_slice(&1000u32.to_be_bytes());
    mvhd.extend_from_slice(&duration_ms.to_be_bytes());

    // `trak` is `tkhd` plus `sinf`. `sinf` is `schm` plus `tenc`.
    let mut tkhd = Vec::new();
    tkhd.extend_from_slice(&1u32.to_be_bytes());
    tkhd.extend_from_slice(&width.to_be_bytes());
    tkhd.extend_from_slice(&height.to_be_bytes());

    let mut schm = Vec::new();
    schm.extend_from_slice(&0u32.to_be_bytes()); // version + flags
    schm.extend_from_slice(b"cenc"); // scheme fourcc
    schm.extend_from_slice(&0x0001_0000u32.to_be_bytes()); // scheme version word

    let mut tenc = Vec::new();
    tenc.extend_from_slice(&0u32.to_be_bytes()); // version + flags
    tenc.push(0); // reserved
    tenc.push(0); // crypt-byte and skip-byte nibble byte, stored as 0
    tenc.push(1); // protected flag. Reader treats a non-zero byte as encrypted.
    tenc.push(16); // IV size byte. The IV itself is carried in `pssh`, not per sample.
    tenc.extend_from_slice(&key_id); // 16-byte key id

    let mut sinf = Vec::new();
    write_mp4_box(&mut sinf, *b"schm", &schm);
    write_mp4_box(&mut sinf, *b"tenc", &tenc);

    let mut trak = Vec::new();
    write_mp4_box(&mut trak, *b"tkhd", &tkhd);
    write_mp4_box(&mut trak, *b"sinf", &sinf);

    // `pssh`: version 1, fixed system id, one key id, then the 16-byte IV used as the CTR counter.
    let mut pssh = Vec::new();
    pssh.extend_from_slice(&0x0100_0000u32.to_be_bytes()); // version 1, flags 0
    pssh.extend_from_slice(&W3C_CLEARKEY_SYSTEM_ID);
    pssh.extend_from_slice(&1u32.to_be_bytes()); // key id count
    pssh.extend_from_slice(&key_id);
    pssh.extend_from_slice(&(iv.len() as u32).to_be_bytes());
    pssh.extend_from_slice(&iv);

    let mut moov = Vec::new();
    write_mp4_box(&mut moov, *b"mvhd", &mvhd);
    write_mp4_box(&mut moov, *b"trak", &trak);
    write_mp4_box(&mut moov, *b"pssh", &pssh);
    write_mp4_box(&mut out, *b"moov", &moov);

    // One AES-128-CTR sample of the whole bitstream. Encrypt and decrypt share `aes128_ctr`.
    let encrypted_mdat = cenc_encrypt_sample(&key, &iv, mdat_plaintext);
    write_mp4_box(&mut out, *b"mdat", &encrypted_mdat);
    out
}

/// Read [`Mp4Metadata`] from `data` without decoding `mdat`.
///
/// The file must be at most [`MAX_CONTAINER_BYTES`], at least 12 bytes, and bytes 4..8
/// must be `ftyp`. The box walk enforces [`MAX_MP4_BOXES`] and [`MAX_MP4_BOX_DEPTH`].
/// Both `moov` and `mdat` must appear. When width and height were both non-zero, they
/// are checked with [`validate_video_dimensions`]. A zero dimension pair is left as read.
pub fn inspect_mp4_metadata(data: &[u8]) -> Result<Mp4Metadata, VideoError> {
    // Reject a file larger than MAX_CONTAINER_BYTES (64 MiB) before walking boxes.
    if data.len() > MAX_CONTAINER_BYTES {
        return Err(VideoError::ContainerTooLarge { size: data.len() });
    }
    if data.len() < 12 || &data[4..8] != b"ftyp" {
        return Err(VideoError::InvalidHeader);
    }
    let mut meta = Mp4Metadata {
        brand: [data[8], data[9], data[10], data[11]],
        width: 0,
        height: 0,
        timescale: 1000,
        duration_ms: 0,
        mdat_size: 0,
        box_count: 0,
        encrypted: false,
        protection_scheme: None,
        default_kid: None,
        iv: None,
        pssh_count: 0,
        has_audio_track: false,
        audio_sample_rate: 0,
        audio_channels: 0,
    };
    let mut saw_moov = false;
    let mut saw_mdat = false;
    scan_mp4_boxes(data, 0, &mut meta, &mut saw_moov, &mut saw_mdat)?;
    if !saw_moov || !saw_mdat {
        return Err(VideoError::InvalidHeader);
    }
    if meta.width > 0 && meta.height > 0 {
        validate_video_dimensions(meta.width, meta.height)?;
    }
    Ok(meta)
}

/// Walk every box in `slice` and fill `meta`.
///
/// Each box starts with a big-endian `u32` size. That size counts the 8-byte header
/// plus the payload. A size below 8, a size that runs past `slice`, and any trailing
/// bytes that are not a whole box are [`VideoError::CorruptedBitstream`]. There is no
/// 64-bit size and no size of zero meaning "extends to the end of the file".
///
/// `depth` is 0 for the file. A call with `depth` greater than [`MAX_MP4_BOX_DEPTH`]
/// fails. More than [`MAX_MP4_BOXES`] counted boxes fails. Container types `moov`,
/// `trak`, `mdia`, `minf`, `stbl`, and `sinf` recurse at `depth + 1`. `moov` sets
/// `saw_moov`. `mdat` sets `saw_mdat`. The caller requires both. This muxer writes
/// `moov` and `mdat`. It does not write `mdia`, `minf`, or `stbl`, but those names
/// are still descended into if a file contains them.
///
/// `mvhd` of at least 12 bytes: timescale at payload offset 4, duration at offset 8.
/// `tkhd` of at least 12 bytes: width at offset 4, height at offset 8.
/// `schm` of at least 8 bytes: four scheme bytes at offset 4.
/// `tenc` of at least 24 bytes: encrypted when payload byte 6 is non-zero, key id at offset 8.
/// `pssh` of at least 20 bytes increments the count. At least 60 bytes also copies the
/// 16-byte IV from offset 44. An `mdat` payload above [`MAX_PACKET_BYTES`] fails.
fn scan_mp4_boxes(
    slice: &[u8],
    depth: usize,
    meta: &mut Mp4Metadata,
    saw_moov: &mut bool,
    saw_mdat: &mut bool,
) -> Result<(), VideoError> {
    // Depth 0 is the file. Depth greater than MAX_MP4_BOX_DEPTH (16) stops the walk.
    if depth > MAX_MP4_BOX_DEPTH {
        return Err(VideoError::BoxDepthExceeded { depth });
    }
    let mut cursor = 0usize;
    while cursor + 8 <= slice.len() {
        meta.box_count += 1;
        // Stop once the running count passes MAX_MP4_BOXES (256).
        if meta.box_count > MAX_MP4_BOXES {
            return Err(VideoError::TooManyBoxes {
                count: meta.box_count,
            });
        }
        let box_size = u32::from_be_bytes([
            slice[cursor],
            slice[cursor + 1],
            slice[cursor + 2],
            slice[cursor + 3],
        ]) as usize;
        // Size includes the 8-byte header and must fit in this slice.
        if box_size < 8 || cursor + box_size > slice.len() {
            return Err(VideoError::CorruptedBitstream);
        }
        let box_type = &slice[cursor + 4..cursor + 8];
        let payload = &slice[cursor + 8..cursor + box_size];
        match box_type {
            b"moov" | b"trak" | b"mdia" | b"minf" | b"stbl" | b"sinf" => {
                if box_type == b"moov" {
                    *saw_moov = true;
                }
                scan_mp4_boxes(payload, depth + 1, meta, saw_moov, saw_mdat)?;
            }
            b"mvhd" if payload.len() >= 12 => {
                meta.timescale =
                    u32::from_be_bytes([payload[4], payload[5], payload[6], payload[7]]).max(1);
                meta.duration_ms =
                    u32::from_be_bytes([payload[8], payload[9], payload[10], payload[11]]);
            }
            b"tkhd" if payload.len() >= 12 => {
                meta.width = u32::from_be_bytes([payload[4], payload[5], payload[6], payload[7]]);
                meta.height =
                    u32::from_be_bytes([payload[8], payload[9], payload[10], payload[11]]);
            }
            b"schm" if payload.len() >= 8 => {
                meta.protection_scheme = Some([payload[4], payload[5], payload[6], payload[7]]);
            }
            b"tenc" if payload.len() >= 24 => {
                let is_protected = payload[6] != 0;
                meta.encrypted = is_protected;
                let mut kid = [0u8; 16];
                kid.copy_from_slice(&payload[8..24]);
                meta.default_kid = Some(kid);
            }
            b"pssh" if payload.len() >= 20 => {
                meta.pssh_count += 1;
                if payload.len() >= 44 + 16 {
                    let mut iv = [0u8; 16];
                    iv.copy_from_slice(&payload[44..60]);
                    meta.iv = Some(iv);
                }
            }
            b"mdat" => {
                // `mdat` payload is capped at MAX_PACKET_BYTES, separate from the 64 MiB file cap.
                if payload.len() > MAX_PACKET_BYTES {
                    return Err(VideoError::PacketTooLarge {
                        size: payload.len(),
                    });
                }
                meta.mdat_size = payload.len();
                *saw_mdat = true;
            }
            b"hdlr" if payload.len() >= 14 => {
                if &payload[4..8] == b"soun" {
                    meta.has_audio_track = true;
                    meta.audio_sample_rate =
                        u32::from_be_bytes([payload[8], payload[9], payload[10], payload[11]]);
                    meta.audio_channels = payload[12];
                }
            }
            b"smhd" if payload.len() >= 8 => {
                meta.has_audio_track = true;
            }
            _ => {}
        }
        cursor += box_size;
    }
    if cursor != slice.len() {
        return Err(VideoError::CorruptedBitstream);
    }
    Ok(())
}

/// Return the payload of the last top-level `mdat` box.
///
/// [`inspect_mp4_metadata`] runs first, so the file must contain `ftyp`, `moov`, and
/// some `mdat`, and it must pass the container, box-count, and depth limits. This
/// second walk does not recurse. The size field is a big-endian `u32` that includes
/// the 8-byte header. An `mdat` payload above [`MAX_PACKET_BYTES`] fails. If several
/// top-level `mdat` boxes exist, the last one is returned. A nested `mdat` alone is
/// not returned, and that yields [`VideoError::InvalidHeader`].
pub fn parse_mp4_container(data: &[u8]) -> Result<&[u8], VideoError> {
    let _meta = inspect_mp4_metadata(data)?;
    let mut cursor = 0usize;
    let mut mdat_slice: Option<&[u8]> = None;

    while cursor + 8 <= data.len() {
        let box_size = u32::from_be_bytes([
            data[cursor],
            data[cursor + 1],
            data[cursor + 2],
            data[cursor + 3],
        ]) as usize;
        // Size includes the 8-byte header and must fit in the file.
        if box_size < 8 || cursor + box_size > data.len() {
            return Err(VideoError::CorruptedBitstream);
        }
        let box_type = &data[cursor + 4..cursor + 8];
        let payload = &data[cursor + 8..cursor + box_size];
        if box_type == b"mdat" {
            // Same 16 MiB payload cap as the nested walk.
            if payload.len() > MAX_PACKET_BYTES {
                return Err(VideoError::PacketTooLarge {
                    size: payload.len(),
                });
            }
            mdat_slice = Some(payload);
        }
        cursor += box_size;
    }

    mdat_slice.ok_or(VideoError::InvalidHeader)
}

/// Append one box: big-endian `u32` size, `fourcc`, then `payload`.
///
/// The size field is `8 + payload.len()` as a `u32` and includes the header.
/// Child boxes are not rewritten here. A caller that builds `moov` passes bytes that
/// were already wrapped. There is no 64-bit largesize form.
fn write_mp4_box(out: &mut Vec<u8>, fourcc: [u8; 4], payload: &[u8]) {
    let total_size = (8 + payload.len()) as u32;
    out.extend_from_slice(&total_size.to_be_bytes());
    out.extend_from_slice(&fourcc);
    out.extend_from_slice(payload);
}

/// Build a short unencrypted demo MP4 of a vertical bar moving across a gradient.
///
/// `frame_count` is clamped to 1..=60. Frame `f` has `pts_ms` of `f * 33`, and only
/// frame 0 has [`VideoFrame::keyframe`] set. Encode still inserts an `I` packet every
/// 8 frames. The bytes come from [`GpuVideoSession::encode_mp4`] at 30 fps, on the CPU.
/// Nothing is sent to a GPU.
pub fn generate_demo_mp4(width: u32, height: u32, frame_count: u32) -> Result<Vec<u8>, VideoError> {
    validate_video_dimensions(width, height)?;
    // Demo clips are clamped to 60 frames even when the caller asks for more.
    let count = frame_count.clamp(1, 60);
    let mut frames = Vec::with_capacity(count as usize);

    for f in 0..count {
        let mut rgba = vec![0u8; (width as usize) * (height as usize) * 4];
        let bar_x = ((f * 6) % width) as i32;
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 4) as usize;
                let dist = ((x as i32) - bar_x).abs();
                if dist < 14 {
                    // Moving gold highlight bar
                    rgba[idx] = 232;
                    rgba[idx + 1] = 195;
                    rgba[idx + 2] = 122;
                    rgba[idx + 3] = 255;
                } else {
                    // Warm dark gradient backdrop
                    rgba[idx] = (24 + (x * 40 / width.max(1))) as u8;
                    rgba[idx + 1] = (32 + (y * 50 / height.max(1))) as u8;
                    rgba[idx + 2] = 56;
                    rgba[idx + 3] = 255;
                }
            }
        }
        frames.push(VideoFrame::new(width, height, f * 33, f == 0, rgba)?);
    }

    let duration_ms = (count * 33).max(100);
    let audio =
        AudioTrack::synthesize_harmonic_chime(duration_ms, 44_100, 2, AudioCodecProfile::ImaAdpcm4);
    let mut session = GpuVideoSession::default();
    session.encode_av_mp4(30, &frames, Some(&audio))
}

/// Build a short demo MP4 whose `mdat` is one AES-128-CTR sample.
///
/// The key id is [`DEFAULT_EME_DEMO_KID`] and the key is [`DEFAULT_EME_DEMO_KEY`].
/// Both are fixed 16-byte arrays in this file, not secrets read from disk. The IV is
/// the fixed 16-byte counter in this function. Encrypt and decrypt share `aes128_ctr`.
/// `frame_count` is clamped to 1..=60. Encode uses [`GpuVideoSession::encode_encrypted_mp4`]
/// at 30 fps. The picture is a vertical bar on a gradient, drawn on the CPU.
pub fn generate_encrypted_demo_mp4(
    width: u32,
    height: u32,
    frame_count: u32,
) -> Result<Vec<u8>, VideoError> {
    validate_video_dimensions(width, height)?;
    // Demo clips are clamped to 60 frames even when the caller asks for more.
    let count = frame_count.clamp(1, 60);
    let mut frames = Vec::with_capacity(count as usize);

    for f in 0..count {
        let mut rgba = vec![0u8; (width as usize) * (height as usize) * 4];
        let bar_x = ((f * 8) % width) as i32;
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 4) as usize;
                let dist = ((x as i32) - bar_x).abs();
                if dist < 12 {
                    // Protected cyan-emerald indicator bar
                    rgba[idx] = 88;
                    rgba[idx + 1] = 196;
                    rgba[idx + 2] = 220;
                    rgba[idx + 3] = 255;
                } else {
                    rgba[idx] = (18 + (x * 36 / width.max(1))) as u8;
                    rgba[idx + 1] = (38 + (y * 44 / height.max(1))) as u8;
                    rgba[idx + 2] = 52;
                    rgba[idx + 3] = 255;
                }
            }
        }
        frames.push(VideoFrame::new(width, height, f * 33, f == 0, rgba)?);
    }

    // Fixed 16-byte demo counter. Not a secret fetched from disk.
    let iv = [
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
        0x10,
    ];
    let mut session = GpuVideoSession::default();
    // Key id and key are the fixed demo constants above, not bytes read from disk.
    session.encode_encrypted_mp4(30, &frames, DEFAULT_EME_DEMO_KID, DEFAULT_EME_DEMO_KEY, iv)
}

#[cfg(test)]
mod tests {
    //! Round-trips for the plain container, the demo key sample, frame residuals, box depth, and dimension caps.

    use super::*;

    #[test]
    fn encrypts_and_decrypts_cenc_mp4_via_w3c_clearkey_eme_session() {
        let encrypted_mp4 = generate_encrypted_demo_mp4(32, 24, 6).unwrap();
        let meta = inspect_mp4_metadata(&encrypted_mp4).unwrap();
        assert!(meta.encrypted);
        assert_eq!(meta.protection_scheme, Some(*b"cenc"));
        assert_eq!(meta.default_kid, Some(DEFAULT_EME_DEMO_KID));
        assert_eq!(meta.pssh_count, 1);

        // 1. Decoding with default W3C ClearKey license succeeds
        let mut session = GpuVideoSession::default();
        assert_eq!(
            session.cdm_session().key_system(),
            EmeKeySystem::W3cClearKey
        );
        let clip = session.decode_video_auto(&encrypted_mp4).unwrap();
        assert_eq!(clip.width, 32);
        assert_eq!(clip.height, 24);
        assert_eq!(clip.frames.len(), 6);
        assert_eq!(session.cdm_session().decrypted_samples(), 1);

        // 2. Removing the license key refuses playback with MissingDecryptionKey
        session.cdm_session_mut().clear_keys();
        assert_eq!(
            session.decode_video_auto(&encrypted_mp4),
            Err(VideoError::MissingDecryptionKey {
                kid: DEFAULT_EME_DEMO_KID
            })
        );
    }

    #[test]
    fn encodes_and_decodes_multi_frame_video_with_keyframes_and_inter_frames() {
        let mp4_bytes = generate_demo_mp4(32, 24, 10).unwrap();
        let mut session = GpuVideoSession::default();
        let clip = session.decode_video_auto(&mp4_bytes).unwrap();
        assert_eq!(clip.width, 32);
        assert_eq!(clip.height, 24);
        assert_eq!(clip.frames.len(), 10);
        assert!(clip.frames[0].keyframe);
        assert!(!clip.frames[1].keyframe);
        assert!(clip.frames[8].keyframe);
        assert_eq!(session.frames_decoded(), 10);
    }

    #[test]
    fn muxes_and_demuxes_isobmff_mp4_video_container() {
        let mp4 = wrap_mp4_container(64, 48, 1000, b"AVID_TEST_STREAM");
        assert_eq!(&mp4[4..8], b"ftyp");
        let meta = inspect_mp4_metadata(&mp4).unwrap();
        assert_eq!(meta.brand, *b"isom");
        assert_eq!(meta.width, 64);
        assert_eq!(meta.height, 48);
        assert_eq!(meta.duration_ms, 1000);
        let extracted = parse_mp4_container(&mp4).unwrap();
        assert_eq!(extracted, b"AVID_TEST_STREAM");
    }

    #[test]
    fn rejects_deeply_nested_mp4_container_atoms_and_supports_frame_seeking() {
        let mp4_bytes = generate_demo_mp4(32, 24, 10).unwrap();
        let mut session = GpuVideoSession::default();
        let clip = session.decode_video_auto(&mp4_bytes).unwrap();
        assert_eq!(clip.keyframe_count(), 2);
        assert_eq!(clip.frame_at_ms(0).unwrap().pts_ms, 0);
        assert_eq!(clip.frame_at_ms(70).unwrap().pts_ms, 66);

        // Build an adversarial 20-level nested `moov` atom chain exceeding MAX_MP4_BOX_DEPTH (16)
        let mut nested = Vec::new();
        write_mp4_box(&mut nested, *b"mvhd", &[0u8; 12]);
        for _ in 0..20 {
            let mut parent = Vec::new();
            write_mp4_box(&mut parent, *b"trak", &nested);
            nested = parent;
        }
        let mut bad_mp4 = Vec::new();
        write_mp4_box(&mut bad_mp4, *b"ftyp", b"isom\x00\x00\x02\x00isommp42");
        write_mp4_box(&mut bad_mp4, *b"moov", &nested);
        write_mp4_box(&mut bad_mp4, *b"mdat", b"AVID");
        assert!(matches!(
            inspect_mp4_metadata(&bad_mp4),
            Err(VideoError::BoxDepthExceeded { .. })
        ));
    }

    #[test]
    fn enforces_4k_dimension_and_packet_security_caps() {
        assert_eq!(
            validate_video_dimensions(8192, 4320),
            Err(VideoError::DimensionsTooLarge {
                width: 8192,
                height: 4320
            })
        );
        assert_eq!(
            validate_video_dimensions(321, 240),
            Err(VideoError::OddDimensions {
                width: 321,
                height: 240
            })
        );
    }

    #[test]
    fn probes_gpu_nv12_hardware_pipeline_and_toggles_acceleration() {
        let mut session = GpuVideoSession::new(true, true);
        assert!(session.backend().is_hardware_accelerated());
        session.set_hardware_acceleration(false, false);
        assert_eq!(session.backend(), GpuVideoBackend::SoftwareFallback);
    }

    #[test]
    fn encode_and_decode_run_plane_conversion_on_the_device() {
        let converts = anvil_gpu::device_converts();
        let presents = anvil_gpu::device_presents();
        let rgba = vec![10u8, 180, 40, 255].repeat(8);
        let frame = VideoFrame::new(4, 2, 0, true, rgba.clone()).unwrap();
        let mut session = GpuVideoSession::new(true, true);
        let bytes = session.encode_avid(30, &[frame]).unwrap();
        let clip = session.decode_avid(&bytes).unwrap();
        assert_eq!(clip.frames.len(), 1);
        assert!(anvil_gpu::device_converts() > converts);
        assert!(anvil_gpu::device_presents() > presents);
        let back = &clip.frames[0].rgba;
        assert_eq!(back.len(), rgba.len());
        assert!((back[0] as i16 - rgba[0] as i16).abs() < 40);
        let mut scaled = vec![0u8; 8 * 4 * 4];
        clip.frames[0]
            .blit_into(&mut scaled, 8, 4, 32, 0, 0, 8, 4, true)
            .unwrap();
        assert_eq!(scaled[3], 255);
    }

    #[test]
    fn encodes_and_decodes_aaud_pcm16_and_ima_adpcm_plus_dual_track_aavm_mp4() {
        let pcm_track =
            AudioTrack::synthesize_harmonic_chime(120, 44_100, 2, AudioCodecProfile::Pcm16Le);
        let aaud_pcm = encode_aaud_bitstream(&pcm_track).unwrap();
        let decoded_pcm = decode_aaud_bitstream(&aaud_pcm).unwrap();
        assert_eq!(decoded_pcm, pcm_track);
        assert!(decoded_pcm.peak_level_at_ms(20, 33) > 10);

        let adpcm_track =
            AudioTrack::synthesize_harmonic_chime(120, 44_100, 2, AudioCodecProfile::ImaAdpcm4);
        let aaud_adpcm = encode_aaud_bitstream(&adpcm_track).unwrap();
        assert!(aaud_adpcm.len() < aaud_pcm.len());
        let decoded_adpcm = decode_aaud_bitstream(&aaud_adpcm).unwrap();
        assert_eq!(decoded_adpcm.samples.len(), adpcm_track.samples.len());
        assert!(decoded_adpcm.peak_level_at_ms(20, 33) > 10);

        let demo_mp4 = generate_demo_mp4(32, 24, 6).unwrap();
        let meta = inspect_mp4_metadata(&demo_mp4).unwrap();
        assert!(meta.has_audio_track);
        assert_eq!(meta.audio_sample_rate, 44_100);
        assert_eq!(meta.audio_channels, 2);

        let mut session = GpuVideoSession::default();
        let clip = session.decode_video_auto(&demo_mp4).unwrap();
        assert!(clip.audio.is_some());
        assert!(clip.peak_audio_level_at_ms(33) > 10);

        let window = pcm_track.slice_pcm_window(0, 50);
        assert!(!window.is_empty());
        let wav = pcm_track.to_wav_bytes();
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        let (backend, _) = probe_os_audio_backend();
        let mut bridge = AudioOutputBridge::new();
        assert_eq!(bridge.backend, backend);
        let bridge_wav = bridge.submit_track_window(&pcm_track, 0, 33, false);
        assert!(bridge_wav > 0);
        assert!(bridge.submitted_samples > 0);
        assert!(bridge.last_peak_level > 10);
    }

    #[test]
    fn encodes_and_decodes_dct_mc_h264_annexb_and_av1_obu_video_streams() {
        let mut frames = Vec::new();
        for f in 0..4u32 {
            let mut rgba = vec![0u8; 16 * 16 * 4];
            for y in 0..16u32 {
                for x in 0..16u32 {
                    let i = ((y * 16 + x) * 4) as usize;
                    rgba[i] = ((x * 12 + f * 8) & 0xFF) as u8;
                    rgba[i + 1] = ((y * 10 + f * 6) & 0xFF) as u8;
                    rgba[i + 2] = 140;
                    rgba[i + 3] = 255;
                }
            }
            frames.push(VideoFrame::new(16, 16, f * 33, f == 0, rgba).unwrap());
        }

        let adct = encode_dct_mc_bitstream(VideoCodecProfile::H264High, 30, 22, &frames).unwrap();
        assert_eq!(&adct[0..4], b"ADCT");
        let clip_adct = decode_dct_mc_bitstream(&adct, true).unwrap();
        assert_eq!(clip_adct.frames.len(), 4);
        assert_eq!(clip_adct.profile, VideoCodecProfile::H264High);

        let h264 = encode_h264_annexb(30, 22, &frames).unwrap();
        assert_eq!(&h264[0..5], b"\x00\x00\x00\x01\x67");
        let mut session = GpuVideoSession::default();
        let clip_h264 = session.decode_video_auto(&h264).unwrap();
        assert_eq!(clip_h264.frames.len(), 4);
        assert_eq!(clip_h264.profile, VideoCodecProfile::H264High);

        let av1 = encode_av1_obu_stream(30, 22, &frames).unwrap();
        assert_eq!(&av1[0..3], b"\x12\x00\x0A");
        let clip_av1 = session.decode_video_auto(&av1).unwrap();
        assert_eq!(clip_av1.frames.len(), 4);
        assert_eq!(clip_av1.profile, VideoCodecProfile::Av1Main);
    }

    #[test]
    fn encodes_and_decodes_lossless_flac_and_perceptual_mdct_audio_streams() {
        let track =
            AudioTrack::synthesize_harmonic_chime(80, 44_100, 2, AudioCodecProfile::FlacLossless);
        let flac_bytes = encode_flac_stream(&track).unwrap();
        assert_eq!(&flac_bytes[0..4], b"fLaC");
        let decoded_flac = decode_flac_stream(&flac_bytes).unwrap();
        assert_eq!(decoded_flac.samples, track.samples);
        assert_eq!(decoded_flac.codec, AudioCodecProfile::FlacLossless);

        let mdct_track =
            AudioTrack::synthesize_harmonic_chime(80, 48_000, 2, AudioCodecProfile::OpusMdct);
        let mdct_bytes = encode_mdct_audio_stream(&mdct_track).unwrap();
        assert_eq!(&mdct_bytes[0..4], b"MDCT");
        let decoded_mdct = decode_mdct_audio_stream(&mdct_bytes).unwrap();
        assert_eq!(decoded_mdct.samples.len(), mdct_track.samples.len());
        assert_eq!(decoded_mdct.codec, AudioCodecProfile::OpusMdct);
        assert!(decoded_mdct.peak_level_at_ms(20, 33) > 5);
    }

    #[test]
    fn muxes_and_demuxes_fmp4_mse_segments_and_webm_ebml_containers() {
        let rgba = vec![80u8, 160, 220, 255].repeat(16 * 16);
        let frame = VideoFrame::new(16, 16, 0, true, rgba).unwrap();
        let av1 = encode_av1_obu_stream(30, 20, std::slice::from_ref(&frame)).unwrap();

        let webm = wrap_webm_container(16, 16, 40, "V_AV1", &av1);
        let webm_meta = inspect_webm_metadata(&webm).unwrap();
        assert_eq!(webm_meta.doc_type, "webm");
        assert_eq!(webm_meta.width, 16);
        assert_eq!(webm_meta.height, 16);
        assert_eq!(webm_meta.video_codec_id, "V_AV1");

        let mut session = GpuVideoSession::default();
        let webm_clip = session.decode_video_auto(&webm).unwrap();
        assert_eq!(webm_clip.frames.len(), 1);

        let init_seg = wrap_fmp4_init_segment(16, 16, VideoCodecProfile::Av1Main, false);
        let media_seg = wrap_fmp4_media_segment(1, 0, &[33], &av1);
        let mut mse = MseSourceBufferEngine::new("video/mp4; codecs=\"av01.0.04M.08\"");
        assert_eq!(mse.append_buffer(&init_seg).unwrap(), 0);
        assert!(mse.init_received());
        assert_eq!(mse.append_buffer(&media_seg).unwrap(), 1);
        assert_eq!(mse.frames().len(), 1);
        assert!(!mse.buffered_ranges().is_empty());
    }

    #[test]
    fn runs_w3c_webcodecs_encoder_decoder_and_media_capabilities_queries() {
        let hw = probe_hardware_video_capabilities();
        assert!(!hw.adapter_description.is_empty());
        assert!(hw.supported_profiles.contains(&VideoCodecProfile::H264High));

        let cap = MediaCapabilitiesInfo::query("video/mp4; codecs=\"avc1.640028\"", 1920, 1080, 60);
        assert!(cap.supported);
        assert!(cap.smooth);

        let mut encoder = WebCodecsVideoEncoder::new();
        encoder.configure("avc1.640028", 16, 16, 30, 22).unwrap();
        let rgba = vec![120u8, 90, 200, 255].repeat(16 * 16);
        let frame = VideoFrame::new(16, 16, 0, true, rgba).unwrap();
        let chunk = encoder.encode(&frame, true).unwrap();
        assert_eq!(chunk.chunk_type, EncodedChunkType::Key);
        assert!(!chunk.data.is_empty());

        let mut decoder = WebCodecsVideoDecoder::new();
        decoder.configure("avc1.640028").unwrap();
        let decoded = decoder.decode(&chunk).unwrap();
        assert_eq!(decoded.width, 16);
        assert_eq!(decoded.height, 16);
    }

    #[test]
    fn encodes_and_decodes_royalty_free_opus_rfc6716_vp8_rfc6386_vp9_and_av1_cdef() {
        // 1. RFC 6716 Opus TOC + Range Decoder + CELT/SILK synthesis
        let track =
            AudioTrack::synthesize_harmonic_chime(60, 48_000, 2, AudioCodecProfile::OpusMdct);
        let opus_pkt = encode_rfc6716_opus_packet(&track).unwrap();
        let toc = OpusPacketHeader::parse(&opus_pkt).unwrap();
        assert_eq!(toc.mode, OpusMode::CeltOnly);
        assert_eq!(toc.bandwidth, OpusBandwidth::Fullband);
        assert_eq!(toc.frame_duration_us, 20_000);
        assert!(toc.stereo);
        assert_eq!(toc.samples_per_frame_48k(), 960);
        let decoded_opus = decode_rfc6716_opus_packet(&opus_pkt).unwrap();
        assert_eq!(decoded_opus.samples.len(), track.samples.len());
        assert_eq!(decoded_opus.sample_rate, 48_000);

        // Also test raw RFC 6716 SILK-mode packet synthesis via OpusRangeDecoder
        let raw_silk_pkt = [0x48u8, 0x55, 0xAA, 0x33, 0xCC, 0x99, 0x11, 0x77];
        let silk_track = decode_rfc6716_opus_packet(&raw_silk_pkt).unwrap();
        assert_eq!(silk_track.sample_rate, 48_000);
        assert_eq!(silk_track.channels, 1);
        assert_eq!(silk_track.samples.len(), 960);

        // 2. RFC 6386 VP8 keyframe start code [0x9D, 0x01, 0x2A], BoolDecoder, WHT/IDCT & loop filter
        let mut rgba = vec![0u8; 16 * 16 * 4];
        for i in 0..(16 * 16) {
            rgba[i * 4] = 90;
            rgba[i * 4 + 1] = 175;
            rgba[i * 4 + 2] = 210;
            rgba[i * 4 + 3] = 255;
        }
        let frame = VideoFrame::new(16, 16, 0, true, rgba).unwrap();
        let vp8_bytes = encode_vp8_rfc6386_stream(30, 18, std::slice::from_ref(&frame)).unwrap();
        assert_eq!(&vp8_bytes[3..6], &[0x9D, 0x01, 0x2A]);
        let vp8_hdr = Vp8FrameHeader::parse(&vp8_bytes).unwrap();
        assert!(vp8_hdr.is_keyframe);
        assert_eq!(vp8_hdr.width, 16);
        assert_eq!(vp8_hdr.height, 16);

        let mut session = GpuVideoSession::default();
        let vp8_clip = session.decode_video_auto(&vp8_bytes).unwrap();
        assert_eq!(vp8_clip.profile, VideoCodecProfile::Vp8Profile0);
        assert_eq!(vp8_clip.frames.len(), 1);

        let mut bool_dec = Vp8BoolDecoder::new(&[0xA5, 0x5A, 0xF0, 0x0F]);
        let _ = bool_dec.read_literal(8);
        let wht = vp8_inverse_wht_4x4(&[64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert!(wht[0] > 0);
        let idct4 = vp8_inverse_idct_4x4(&[128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert!(idct4[0] > 0);
        let (p0, q0) = vp8_loop_filter_edge(100, 104, 112, 116, 40);
        assert!(p0 >= 104 && q0 <= 112);

        // 3. VP9 sync code [0x49, 0x83, 0x42] & Annex-B superframe index
        let vp9_bytes = encode_vp9_stream(30, 18, std::slice::from_ref(&frame)).unwrap();
        assert_eq!(&vp9_bytes[1..4], &[0x49, 0x83, 0x42]);
        let vp9_clip = session.decode_video_auto(&vp9_bytes).unwrap();
        assert_eq!(vp9_clip.profile, VideoCodecProfile::Vp9Profile0);
        assert_eq!(vp9_clip.frames.len(), 1);

        // Superframe index for 2 subframes of 4 bytes and 6 bytes (marker 0xC1 = 0b110_00_001)
        let superframe = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0xC1, 4, 6, 0xC1];
        assert_eq!(parse_vp9_superframe_index(&superframe), Some(vec![4, 6]));

        // 4. AV1 LEB128 & CDEF directional filter
        assert_eq!(read_obu_leb128(&[0x85, 0x01]).unwrap(), (133, 2));
        let mut block4 = [
            100u8, 102, 100, 102, 102, 108, 102, 100, 100, 102, 100, 102, 102, 100, 102, 100,
        ];
        av1_cdef_filter_4x4(&mut block4, 8, 4);
        assert!(block4[5] <= 108);
    }
}
