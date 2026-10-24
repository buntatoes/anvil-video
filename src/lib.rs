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
//! `compress_zlib` then wraps that payload. Those packets form an `FVID` bitstream,
//! and that bitstream is the `mdat` body. Decode calls `decompress_zlib_bounded`
//! and `Nv12Surface::to_rgba_bt709`.
//!
//! Sample encryption uses a 16-byte AES-128 key and a 16-byte IV, which is the
//! initial CTR counter. [`cenc_encrypt_sample`] and [`cenc_decrypt_sample`] both
//! call `aes128_ctr`. The whole `FVID` buffer is one sample. There are no per-frame
//! subsample ranges.
//!
//! Three sibling crates are called, and only for these calls:
//! - `root-compress`: `compress_zlib` on encode, `decompress_zlib_bounded` on decode.
//! - `root-cipher`: `aes128_ctr` from both sample functions above.
//! - `root-gpu`: `Nv12Surface::from_rgba_bt709` and `Nv12Surface::to_rgba_bt709`,
//!   which convert color in memory on the CPU.
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

use root_cipher::aes128_ctr;
use root_compress::{CompressError, compress_zlib, decompress_zlib_bounded};
use root_gpu::Nv12Surface;

/// Largest accepted frame width, in pixels.
pub const MAX_VIDEO_WIDTH: u32 = 4096;

/// Largest accepted frame height, in pixels.
pub const MAX_VIDEO_HEIGHT: u32 = 2160;

/// Largest accepted pixel count for one frame (`4096 * 2160`).
pub const MAX_VIDEO_PIXELS: u64 = 8_847_360;

/// Largest accepted frame count in one `FVID` bitstream.
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
    0x46, 0x4c, 0x55, 0x58, 0x5f, 0x43, 0x45, 0x4e, 0x43, 0x5f, 0x4b, 0x49, 0x44, 0x5f, 0x30, 0x31,
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

/// Failure from a limit check, a box walk, an `FVID` parse, or zlib.
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
    /// `root-compress` rejected the zlib packet.
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

/// Four-byte label. This crate does not decode H.264 or AV1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoCodecProfile {
    /// Label whose [`Self::fourcc`] is `avc1`. Not an H.264 decoder.
    H264High,
    /// Label whose [`Self::fourcc`] is `av01`. Not an AV1 decoder.
    Av1Main,
    /// Label for the `FVID` bitstream this crate reads and writes. [`Self::fourcc`] is `fvid`.
    RootNv12Temporal,
}

impl VideoCodecProfile {
    /// `avc1`, `av01`, or `fvid`. The muxer does not write this code into a sample entry.
    ///
    /// The bitstream magic is the separate four bytes `FVID`.
    pub fn fourcc(self) -> [u8; 4] {
        match self {
            VideoCodecProfile::H264High => *b"avc1",
            VideoCodecProfile::Av1Main => *b"av01",
            VideoCodecProfile::RootNv12Temporal => *b"fvid",
        }
    }
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
}

/// Frames decoded from one `FVID` bitstream, plus the header fields copied next to them.
///
/// `backend` is a label from [`probe_platform_gpu_backend`], not a measurement from a device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoClip {
    /// Width copied from the `FVID` header.
    pub width: u32,
    /// Height copied from the `FVID` header.
    pub height: u32,
    /// Frames per second copied from the `FVID` header, at least 1.
    pub fps: u32,
    /// Backend label stored by the decoder. Not a device handle.
    pub backend: GpuVideoBackend,
    /// Profile stored by the decoder. [`GpuVideoSession::decode_fvid`] writes
    /// [`VideoCodecProfile::RootNv12Temporal`].
    pub profile: VideoCodecProfile,
    /// Frames in bitstream order.
    pub frames: Vec<VideoFrame>,
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
    /// After [`GpuVideoSession::decode_fvid`], that flag is set on `I` packets only.
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
}

/// Label stored beside the demo key list. Selecting a variant does not load a module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmeKeySystem {
    /// Label `org.w3.clearkey`. [`CdmSession::with_default_demo_key`] uses this.
    /// No key-system handshake is run.
    W3cClearKey,
    /// Label `org.root.mediafoundation.cdm`. No operating-system media API is called.
    OsMediaFoundationDrm,
    /// Label `org.root.external.cdm`. No plugin file is opened.
    ExternalPluginCdm,
}

impl EmeKeySystem {
    /// Stable identifier string for this label.
    pub fn identifier(self) -> &'static str {
        match self {
            EmeKeySystem::W3cClearKey => "org.w3.clearkey",
            EmeKeySystem::OsMediaFoundationDrm => "org.root.mediafoundation.cdm",
            EmeKeySystem::ExternalPluginCdm => "org.root.external.cdm",
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

/// CPU encode and decode state for `FVID` and for the small MP4 wrapper around it.
///
/// The name says GPU. What it stores is a [`GpuVideoBackend`] label, two boolean flags,
/// a count of frames passed through encode, a count of frames passed through decode,
/// and a [`CdmSession`] that starts with the fixed demo key. Encode and decode convert
/// NV12 in memory with `root-gpu` and compress with `root-compress`. They do not open
/// a device, and the flags do not change the plane math. Either flag updates the stored
/// backend label when the session is created or when [`GpuVideoSession::set_hardware_acceleration`]
/// runs. The encode flag also writes the `FVID` flag byte. The decode flag is what
/// [`GpuVideoSession::decode_fvid`] passes to [`probe_platform_gpu_backend`] for the clip.
#[derive(Clone, Debug)]
pub struct GpuVideoSession {
    /// Label from [`probe_platform_gpu_backend`]. Not a device handle.
    backend: GpuVideoBackend,
    /// Stored flag. Decode still runs on the CPU.
    hardware_decode_enabled: bool,
    /// Stored flag. Encode still runs on the CPU. Also selects the `FVID` flag byte.
    hardware_encode_enabled: bool,
    /// Frames emitted by [`Self::decode_fvid`].
    frames_decoded: u64,
    /// Frames accepted by [`Self::encode_fvid`].
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

    /// Frames emitted by [`Self::decode_fvid`] on this session.
    ///
    /// The count advances as each frame is pushed. A later error in that call does not
    /// subtract frames already counted.
    pub fn frames_decoded(&self) -> u64 {
        self.frames_decoded
    }

    /// Frames appended by [`Self::encode_fvid`] on this session.
    ///
    /// The count advances as each frame is written. A later error in that call does not
    /// subtract frames already counted.
    pub fn frames_encoded(&self) -> u64 {
        self.frames_encoded
    }

    /// Write an `FVID` bitstream for `frames`.
    ///
    /// The header is big-endian: magic `FVID`, version `u16` 1, one flag byte, one zero
    /// byte, width, height, `fps` (stored as at least 1), and the frame count. The flag
    /// byte is 1 only when the encode flag is set and [`GpuVideoBackend::is_hardware_accelerated`]
    /// is true for the stored label. Decode does not read that byte. Planes are still built
    /// on the CPU.
    ///
    /// A frame is marked `I` when its index is 0, [`VideoFrame::keyframe`] is set, or the
    /// index is a multiple of 8. An `I` packet, or a packet with no previous planes, is the
    /// Y plane followed by the UV plane from `Nv12Surface::from_rgba_bt709`. Any other frame
    /// is `P`: each plane byte minus the previous plane byte with `wrapping_sub`. There is
    /// no motion search. `compress_zlib` from `root-compress` wraps that buffer. The record
    /// is the type byte, `pts_ms`, the compressed length, and the compressed bytes.
    /// Every frame must match the first frame's dimensions. An empty slice is
    /// [`VideoError::InvalidHeader`]. `frames_encoded` grows by one per written frame.
    pub fn encode_fvid(&mut self, fps: u32, frames: &[VideoFrame]) -> Result<Vec<u8>, VideoError> {
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
        out.extend_from_slice(b"FVID");
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
            let curr_nv12 = frame.to_nv12_surface()?;
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

    /// Read an `FVID` bitstream into a [`VideoClip`].
    ///
    /// Requires magic `FVID` and version 1. The flag byte at offset 6 is not read.
    /// An `I` packet replaces the reference Y and UV planes. A `P` packet adds each
    /// residual byte with `wrapping_add`. The first frame must be `I`. Packet bytes
    /// are inflated with `decompress_zlib_bounded`, capped at the plane size, then
    /// turned back into RGBA with [`VideoFrame::from_nv12_surface`]. The clip profile
    /// is [`VideoCodecProfile::RootNv12Temporal`]. `backend` is
    /// [`probe_platform_gpu_backend`] of the stored decode flag, not a device result.
    /// `frames_decoded` grows by one per emitted frame.
    pub fn decode_fvid(&mut self, data: &[u8]) -> Result<VideoClip, VideoError> {
        // Magic through frame count is 24 bytes. Offset 6 (flag) and 7 (reserved) are unread.
        if data.len() < 24 || &data[0..4] != b"FVID" {
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

            let decoded_frame = VideoFrame::from_nv12_surface(&ref_surface, pts_ms, is_keyframe)?;
            frames.push(decoded_frame);
            self.frames_decoded += 1;
        }

        let active_backend = probe_platform_gpu_backend(self.hardware_decode_enabled);
        Ok(VideoClip {
            width,
            height,
            fps,
            backend: active_backend,
            profile: VideoCodecProfile::RootNv12Temporal,
            frames,
        })
    }

    /// Encode `frames` as `FVID`, then wrap that bitstream with [`wrap_mp4_container`].
    ///
    /// Duration written into `mvhd` is the last frame's `pts_ms`, or 0 when that is missing.
    /// [`Self::encode_fvid`] already rejects an empty list, so the first frame exists here.
    pub fn encode_mp4(&mut self, fps: u32, frames: &[VideoFrame]) -> Result<Vec<u8>, VideoError> {
        let fvid_stream = self.encode_fvid(fps, frames)?;
        let width = frames[0].width;
        let height = frames[0].height;
        let duration_ms = frames.last().map(|f| f.pts_ms).unwrap_or(0);
        Ok(wrap_mp4_container(width, height, duration_ms, &fvid_stream))
    }

    /// Encode `frames` as `FVID`, then wrap that bitstream as one AES-128-CTR sample.
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
        let fvid_stream = self.encode_fvid(fps, frames)?;
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
            &fvid_stream,
        ))
    }

    /// Decode a buffer that is either an MP4 starting with `ftyp` or a raw `FVID` bitstream.
    ///
    /// For `ftyp`, metadata is read and the top-level `mdat` payload is taken. When `tenc`
    /// says the sample is protected, the 16-byte key id (or [`DEFAULT_EME_DEMO_KID`] when
    /// the file has none) is looked up in the in-memory list. The IV is the 16-byte `pssh`
    /// value, or 16 zero bytes. [`cenc_decrypt_sample`] runs `aes128_ctr` on the whole
    /// payload. One successful call increments the decrypt counter by one. A missing key
    /// returns [`VideoError::MissingDecryptionKey`] and does not decode. A wrong key is
    /// not reported here. The following `FVID` parse fails if the plaintext is not valid.
    /// Anything that is not an `ftyp` buffer is passed straight to [`Self::decode_fvid`].
    pub fn decode_video_auto(&mut self, data: &[u8]) -> Result<VideoClip, VideoError> {
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
                return self.decode_fvid(&decrypted);
            }
            return self.decode_fvid(payload);
        }
        self.decode_fvid(data)
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

    let mut session = GpuVideoSession::default();
    session.encode_mp4(30, &frames)
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
        let mp4 = wrap_mp4_container(64, 48, 1000, b"FVID_TEST_STREAM");
        assert_eq!(&mp4[4..8], b"ftyp");
        let meta = inspect_mp4_metadata(&mp4).unwrap();
        assert_eq!(meta.brand, *b"isom");
        assert_eq!(meta.width, 64);
        assert_eq!(meta.height, 48);
        assert_eq!(meta.duration_ms, 1000);
        let extracted = parse_mp4_container(&mp4).unwrap();
        assert_eq!(extracted, b"FVID_TEST_STREAM");
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
        write_mp4_box(&mut bad_mp4, *b"mdat", b"FVID");
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
}
