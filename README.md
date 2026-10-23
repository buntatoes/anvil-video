# flux-video

[![Release](https://img.shields.io/badge/release-v.03-e8c37a)](https://github.com/buntatoes/flux-video/releases/tag/v.03)
[![License: MPL-2.0](https://img.shields.io/badge/license-MPL--2.0-blue.svg)](./LICENSE)
[![Edition](https://img.shields.io/badge/rust-2024_edition-orange.svg)](https://github.com/buntatoes/flux-video)

`flux-video` is the standalone GPU-accelerated `NV12`/BT.709 video encoder, decoder, ISO Base Media File Format (`ISOBMFF` / `.mp4`) container, and ISO/IEC 23001-7 Common Encryption (`CENC`) + W3C Encrypted Media Extensions (`EME` / `CDM`) library for the Flux rendering engine.

---

## Features (`v.03`)

- **ISO/IEC 23001-7 Common Encryption (`CENC`) & W3C Encrypted Media Extensions (`EME` / `CdmSession`)**:
  - Muxes, inspects, and decrypts `cenc`-protected `.mp4` containers (`sinf`, `schm`, `tenc`, and W3C Common `pssh` boxes) via `wrap_encrypted_mp4_container`, `encode_encrypted_mp4`, `cenc_encrypt_sample`, `cenc_decrypt_sample`, and `generate_encrypted_demo_mp4`.
  - Built-in W3C ClearKey (`EmeKeySystem::W3cClearKey` / `"org.w3.clearkey"`), OS Media Foundation / DRM Bridge (`EmeKeySystem::OsMediaFoundationDrm`), and External CDM Host Plugin Slot (`EmeKeySystem::ExternalPluginCdm`) session management (`CdmSession`) with `(KeyId, ContentKey)` license provisioning and automatic `mdat` decryption inside `GpuVideoSession::decode_video_auto`.
- **GPU Hardware Video Decode & Encode Session (`GpuVideoSession`, `GpuVideoBackend`)**:
  - Coordinates semi-planar 4:2:0 `NV12` GPU video surfaces (`flux-gpu::Nv12Surface`) across **Windows Direct3D 11 Video / Media Foundation DXVA2** (`GpuVideoBackend::D3d11Dxva2` — NVIDIA NVDEC/NVENC, AMD AMF/VCN, Intel QuickSync) and **Linux VA-API / DRM** (`GpuVideoBackend::LinuxVaApi`), with runtime hardware decode/encode toggling and automatic fallback to `GpuVideoBackend::SoftwareFallback`.
- **Temporal `NV12` Keyframe (`I`) & Inter-Frame (`P`) Video Codec (`encode_fvid`, `decode_fvid`)**:
  - Encodes `RGBA8888` video frames into BT.709 `NV12` luma (`Y`) and chroma (`UV`) planes, emitting periodic `I-frames` (keyframes) and temporal residual `P-frames` compressed via bounded `flux-compress` zlib streams, with timestamp frame seeking (`VideoClip::frame_at_ms`) and keyframe telemetry (`VideoClip::keyframe_count`).
- **ISOBMFF / `.mp4` Container Muxer, Demuxer, & Metadata Inspector (`wrap_mp4_container`, `parse_mp4_container`, `inspect_mp4_metadata`, `decode_video_auto`)**:
  - Writes and parses standard ISO Base Media File Format (`ftyp`, `moov`, `mvhd`, `trak`, `tkhd`, `sinf`, `schm`, `tenc`, `pssh`, `mdat`) container boxes, inspects header dimensions/timescale/duration and encryption scheme (`Mp4Metadata`) prior to bitstream decoding, and automatically detects `.mp4` or raw `FVID` video streams.
- **Strict 4K, Container, & Atom-Bomb Security Caps (`validate_video_dimensions`, `inspect_mp4_metadata`)**:
  - Enforces `MAX_VIDEO_WIDTH = 4096`, `MAX_VIDEO_HEIGHT = 2160` (4K ceiling), `MAX_VIDEO_FRAMES = 36,000`, `MAX_PACKET_BYTES = 16 MiB`, `MAX_CONTAINER_BYTES = 64 MiB`, `MAX_MP4_BOXES = 256`, and `MAX_MP4_BOX_DEPTH = 16` prior to buffer allocation.

---

## Quick Start

```rust
use flux_video::{GpuVideoSession, generate_demo_mp4};

let mp4_bytes = generate_demo_mp4(64, 36, 12).unwrap();
let mut session = GpuVideoSession::default();
let clip = session.decode_video_auto(&mp4_bytes).unwrap();

assert_eq!(clip.width, 64);
assert_eq!(clip.height, 36);
assert_eq!(clip.frames.len(), 12);
```

---

## Building and Testing

```powershell
cargo test
```

---

## License

Licensed under the Mozilla Public License Version 2.0 ([`MPL-2.0`](./LICENSE)).

## Author

**Buntos**
