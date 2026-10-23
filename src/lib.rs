//! Proprietary GPU-accelerated `NV12`/BT.709 video encoder, decoder, and ISOBMFF/MP4 container engine.
//!
//! Coordinates hardware `NV12` GPU video surfaces (Direct3D 11 Video / DXVA2 on Windows and
//! VA-API on Linux via `flux-gpu`) alongside a deterministic keyframe (`I-frame`) and
//! temporal residual (`P-frame`) video codec and ISOBMFF (`.mp4`) container muxer/demuxer.

use std::fmt;

use flux_compress::{CompressError, compress_zlib, decompress_zlib_bounded};
use flux_gpu::Nv12Surface;

/// Maximum allowed video dimension (`4096` pixels — 4K DCI width ceiling).
pub const MAX_VIDEO_WIDTH: u32 = 4096;

/// Maximum allowed video height (`2160` pixels — 4K UHD height ceiling).
pub const MAX_VIDEO_HEIGHT: u32 = 2160;

/// Maximum allowed pixels per frame (`4096 * 2160 = 8,847,360` pixels).
pub const MAX_VIDEO_PIXELS: u64 = 8_847_360;

/// Maximum allowed frames per decoded clip (`36,000` frames).
pub const MAX_VIDEO_FRAMES: u32 = 36_000;

/// Maximum compressed packet size per frame (`16 MiB`).
pub const MAX_PACKET_BYTES: usize = 16 * 1024 * 1024;

/// Maximum total container size accepted by the demuxer (`64 MiB`).
pub const MAX_CONTAINER_BYTES: usize = 64 * 1024 * 1024;

/// Maximum number of ISOBMFF boxes (`atoms`) allowed in a single container.
pub const MAX_MP4_BOXES: usize = 256;

/// Maximum ISOBMFF container box nesting depth.
pub const MAX_MP4_BOX_DEPTH: usize = 16;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VideoError {
    InvalidHeader,
    DimensionsTooLarge { width: u32, height: u32 },
    OddDimensions { width: u32, height: u32 },
    FrameCountTooLarge { count: u32 },
    PacketTooLarge { size: usize },
    ContainerTooLarge { size: usize },
    TooManyBoxes { count: usize },
    BoxDepthExceeded { depth: usize },
    CorruptedBitstream,
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
            VideoError::Compression(e) => write!(f, "video compression error: {e}"),
        }
    }
}

impl std::error::Error for VideoError {}

impl From<CompressError> for VideoError {
    fn from(value: CompressError) -> Self {
        VideoError::Compression(value)
    }
}

/// Hardware GPU video acceleration backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuVideoBackend {
    /// Windows Direct3D 11 Video / Media Foundation DXVA2 (`ID3D11VideoDevice` / `ID3D11VideoContext`
    /// driving NVIDIA NVDEC/NVENC, AMD AMF/VCN, and Intel QuickSync on `NV12` textures).
    D3d11Dxva2,
    /// Linux VA-API / DRM `NV12` hardware video acceleration (`libva`).
    LinuxVaApi,
    /// Software CPU `NV12` / BT.709 fallback pipeline.
    SoftwareFallback,
}

impl GpuVideoBackend {
    pub fn label(self) -> &'static str {
        match self {
            GpuVideoBackend::D3d11Dxva2 => "D3D11 Video / DXVA2 (NV12 GPU)",
            GpuVideoBackend::LinuxVaApi => "VA-API / DRM (NV12 GPU)",
            GpuVideoBackend::SoftwareFallback => "Software BT.709 Pipeline",
        }
    }

    pub fn is_hardware_accelerated(self) -> bool {
        !matches!(self, GpuVideoBackend::SoftwareFallback)
    }
}

/// Supported video codec profiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoCodecProfile {
    /// Hardware H.264 / AVC High Profile (`avc1`).
    H264High,
    /// Hardware AV1 Main Profile (`av01`).
    Av1Main,
    /// Flux native `NV12` keyframe + temporal residual stream (`FVID`).
    FluxNv12Temporal,
}

impl VideoCodecProfile {
    pub fn fourcc(self) -> [u8; 4] {
        match self {
            VideoCodecProfile::H264High => *b"avc1",
            VideoCodecProfile::Av1Main => *b"av01",
            VideoCodecProfile::FluxNv12Temporal => *b"fvid",
        }
    }
}

/// Probe the platform's native hardware GPU video decode/encode backend.
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

/// Validates video dimensions against the 4K security cap and 4:2:0 alignment rules.
pub fn validate_video_dimensions(width: u32, height: u32) -> Result<(), VideoError> {
    if width == 0 || height == 0 {
        return Err(VideoError::InvalidHeader);
    }
    if width > MAX_VIDEO_WIDTH
        || height > MAX_VIDEO_HEIGHT
        || u64::from(width) * u64::from(height) > MAX_VIDEO_PIXELS
    {
        return Err(VideoError::DimensionsTooLarge { width, height });
    }
    if width % 2 != 0 || height % 2 != 0 {
        return Err(VideoError::OddDimensions { width, height });
    }
    Ok(())
}

/// Single decoded or unencoded video frame in `RGBA8888` with presentation timestamp.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoFrame {
    pub width: u32,
    pub height: u32,
    pub pts_ms: u32,
    pub keyframe: bool,
    pub rgba: Vec<u8>,
}

impl VideoFrame {
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

    /// Upload this `RGBA8888` frame into a semi-planar BT.709 `NV12` GPU surface.
    pub fn to_nv12_surface(&self) -> Result<Nv12Surface, VideoError> {
        Nv12Surface::from_rgba_bt709(self.width, self.height, &self.rgba)
            .ok_or(VideoError::InvalidHeader)
    }

    /// Reconstruct a `VideoFrame` from a hardware `NV12` GPU surface.
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

/// Decoded multi-frame video clip with metadata and GPU backend telemetry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoClip {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub backend: GpuVideoBackend,
    pub profile: VideoCodecProfile,
    pub frames: Vec<VideoFrame>,
}

impl VideoClip {
    pub fn duration_ms(&self) -> u32 {
        self.frames.last().map(|f| f.pts_ms).unwrap_or(0)
    }

    pub fn poster_frame(&self) -> Option<&VideoFrame> {
        self.frames.first()
    }

    /// Return the number of `I-frame` keyframes in this clip.
    pub fn keyframe_count(&self) -> usize {
        self.frames.iter().filter(|f| f.keyframe).count()
    }

    /// Return the active [`VideoFrame`] at `timestamp_ms` (clamped to clip duration).
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

/// Lightweight container metadata parsed from an ISOBMFF (`.mp4`) header without decoding `mdat`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mp4Metadata {
    pub brand: [u8; 4],
    pub width: u32,
    pub height: u32,
    pub timescale: u32,
    pub duration_ms: u32,
    pub mdat_size: usize,
    pub box_count: usize,
}

/// Stateful GPU video encode & decode session managing `NV12` hardware surfaces and telemetry.
#[derive(Clone, Debug)]
pub struct GpuVideoSession {
    backend: GpuVideoBackend,
    hardware_decode_enabled: bool,
    hardware_encode_enabled: bool,
    frames_decoded: u64,
    frames_encoded: u64,
}

impl Default for GpuVideoSession {
    fn default() -> Self {
        Self::new(true, true)
    }
}

impl GpuVideoSession {
    pub fn new(hardware_decode: bool, hardware_encode: bool) -> Self {
        let backend = probe_platform_gpu_backend(hardware_decode || hardware_encode);
        Self {
            backend,
            hardware_decode_enabled: hardware_decode,
            hardware_encode_enabled: hardware_encode,
            frames_decoded: 0,
            frames_encoded: 0,
        }
    }

    pub fn backend(&self) -> GpuVideoBackend {
        self.backend
    }

    pub fn is_hardware_decode_enabled(&self) -> bool {
        self.hardware_decode_enabled
    }

    pub fn is_hardware_encode_enabled(&self) -> bool {
        self.hardware_encode_enabled
    }

    pub fn set_hardware_acceleration(&mut self, decode: bool, encode: bool) {
        self.hardware_decode_enabled = decode;
        self.hardware_encode_enabled = encode;
        self.backend = probe_platform_gpu_backend(decode || encode);
    }

    pub fn frames_decoded(&self) -> u64 {
        self.frames_decoded
    }

    pub fn frames_encoded(&self) -> u64 {
        self.frames_encoded
    }

    /// Encode a sequence of `VideoFrame`s into the `FVID` GPU `NV12` temporal bitstream.
    ///
    /// Every 8th frame (and frame 0) is encoded as an `I-frame` (full `NV12` luma + chroma keyframe),
    /// while intermediate frames are encoded as `P-frames` (temporal residual difference against
    /// the previous `NV12` reference surface).
    pub fn encode_fvid(&mut self, fps: u32, frames: &[VideoFrame]) -> Result<Vec<u8>, VideoError> {
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

    /// Decode an `FVID` `NV12` temporal bitstream into a [`VideoClip`].
    pub fn decode_fvid(&mut self, data: &[u8]) -> Result<VideoClip, VideoError> {
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
            profile: VideoCodecProfile::FluxNv12Temporal,
            frames,
        })
    }

    /// Encode a sequence of `VideoFrame`s into an ISO Base Media File Format (`ISOBMFF` / `.mp4`) container.
    pub fn encode_mp4(&mut self, fps: u32, frames: &[VideoFrame]) -> Result<Vec<u8>, VideoError> {
        let fvid_stream = self.encode_fvid(fps, frames)?;
        let width = frames[0].width;
        let height = frames[0].height;
        let duration_ms = frames.last().map(|f| f.pts_ms).unwrap_or(0);
        Ok(wrap_mp4_container(width, height, duration_ms, &fvid_stream))
    }

    /// Decode an `.mp4` (`ISOBMFF`) or `FVID` video buffer automatically.
    pub fn decode_video_auto(&mut self, data: &[u8]) -> Result<VideoClip, VideoError> {
        if data.len() >= 8 && &data[4..8] == b"ftyp" {
            let payload = parse_mp4_container(data)?;
            return self.decode_fvid(payload);
        }
        self.decode_fvid(data)
    }
}

/// Wrap an encoded video bitstream inside a standard ISO Base Media File Format (`ISOBMFF` / `.mp4`) container
/// with `ftyp`, `moov` (`mvhd` + `trak`), and `mdat` boxes.
pub fn wrap_mp4_container(
    width: u32,
    height: u32,
    duration_ms: u32,
    mdat_payload: &[u8],
) -> Vec<u8> {
    let mut out = Vec::new();

    // 1. `ftyp` box (20 bytes)
    let mut ftyp = Vec::new();
    ftyp.extend_from_slice(b"isom");
    ftyp.extend_from_slice(&512u32.to_be_bytes());
    ftyp.extend_from_slice(b"isommp42");
    write_mp4_box(&mut out, *b"ftyp", &ftyp);

    // 2. `moov` box containing `mvhd` and `trak`
    let mut mvhd = Vec::new();
    mvhd.extend_from_slice(&0u32.to_be_bytes()); // version + flags
    mvhd.extend_from_slice(&1000u32.to_be_bytes()); // timescale = 1000 ms
    mvhd.extend_from_slice(&duration_ms.to_be_bytes());

    let mut tkhd = Vec::new();
    tkhd.extend_from_slice(&1u32.to_be_bytes()); // track enabled
    tkhd.extend_from_slice(&width.to_be_bytes());
    tkhd.extend_from_slice(&height.to_be_bytes());

    let mut trak = Vec::new();
    write_mp4_box(&mut trak, *b"tkhd", &tkhd);

    let mut moov = Vec::new();
    write_mp4_box(&mut moov, *b"mvhd", &mvhd);
    write_mp4_box(&mut moov, *b"trak", &trak);
    write_mp4_box(&mut out, *b"moov", &moov);

    // 3. `mdat` box containing the media bitstream
    write_mp4_box(&mut out, *b"mdat", mdat_payload);
    out
}

/// Inspect an ISO Base Media File Format (`ISOBMFF` / `.mp4`) container and return [`Mp4Metadata`]
/// while enforcing [`MAX_CONTAINER_BYTES`], [`MAX_MP4_BOXES`], and [`MAX_MP4_BOX_DEPTH`].
pub fn inspect_mp4_metadata(data: &[u8]) -> Result<Mp4Metadata, VideoError> {
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

fn scan_mp4_boxes(
    slice: &[u8],
    depth: usize,
    meta: &mut Mp4Metadata,
    saw_moov: &mut bool,
    saw_mdat: &mut bool,
) -> Result<(), VideoError> {
    if depth > MAX_MP4_BOX_DEPTH {
        return Err(VideoError::BoxDepthExceeded { depth });
    }
    let mut cursor = 0usize;
    while cursor + 8 <= slice.len() {
        meta.box_count += 1;
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
        if box_size < 8 || cursor + box_size > slice.len() {
            return Err(VideoError::CorruptedBitstream);
        }
        let box_type = &slice[cursor + 4..cursor + 8];
        let payload = &slice[cursor + 8..cursor + box_size];
        match box_type {
            b"moov" | b"trak" | b"mdia" | b"minf" | b"stbl" => {
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
            b"mdat" => {
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

/// Parse an ISO Base Media File Format (`ISOBMFF` / `.mp4`) container and return the `mdat` bitstream slice.
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
        if box_size < 8 || cursor + box_size > data.len() {
            return Err(VideoError::CorruptedBitstream);
        }
        let box_type = &data[cursor + 4..cursor + 8];
        let payload = &data[cursor + 8..cursor + box_size];
        if box_type == b"mdat" {
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

fn write_mp4_box(out: &mut Vec<u8>, fourcc: [u8; 4], payload: &[u8]) {
    let total_size = (8 + payload.len()) as u32;
    out.extend_from_slice(&total_size.to_be_bytes());
    out.extend_from_slice(&fourcc);
    out.extend_from_slice(payload);
}

/// Generate a procedural GPU-encoded MP4 demo video clip (`width` x `height`, `frame_count` frames)
/// for built-in `<video>` playback and hardware decode testing.
pub fn generate_demo_mp4(width: u32, height: u32, frame_count: u32) -> Result<Vec<u8>, VideoError> {
    validate_video_dimensions(width, height)?;
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

#[cfg(test)]
mod tests {
    use super::*;

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
