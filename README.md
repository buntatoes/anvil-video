<div align="center">
  <h1>flux-video</h1>
  <p><strong>GPU-Accelerated NV12/BT.709 Video Codec, ISOBMFF/MP4 Container Engine &amp; ISO/IEC 23001-7 CENC / W3C EME Session (Proof of Concept)</strong></p>
  <p>
    <a href="https://github.com/buntatoes/flux-video/releases/tag/v.03"><img src="https://img.shields.io/badge/release-v.03-e8c37a?style=flat-square" alt="Release v.03" /></a>
    <img src="https://img.shields.io/badge/status-Proof_of_Concept_(PoC)-d4a359?style=flat-square" alt="Proof of Concept" />
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MPL--2.0-blue?style=flat-square" alt="License MPL-2.0" /></a>
    <a href="./SECURITY.md"><img src="https://img.shields.io/badge/security-policy-2ea44f?style=flat-square" alt="Security Policy" /></a>
    <a href="https://github.com/buntatoes/flux-video"><img src="https://img.shields.io/badge/rust-2024_edition-orange?style=flat-square" alt="Rust 2024 Edition" /></a>
  </p>
</div>

> [!NOTE]
> **Proof of Concept (PoC) Notice**: `flux-video` is a standalone **Proof of Concept** GPU-accelerated `NV12`/BT.709 video encoder, decoder, ISO Base Media File Format (`ISOBMFF` / `.mp4`) container muxer/demuxer, and ISO/IEC 23001-7 Common Encryption (`CENC`) + W3C Encrypted Media Extensions (`EME` / `CDM`) library authored from scratch by **Buntos** as part of the [**Flux Browser**](https://github.com/buntatoes/flux-browser) engine ecosystem.

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

## API Overview

| Type / Function | Description |
| :--- | :--- |
| `GpuVideoSession` | Hardware-aware `NV12` video decode & encode session (`D3d11Dxva2`, `LinuxVaApi`, `SoftwareFallback`). |
| `GpuVideoSession::decode_video_auto(&[u8]) -> Result<VideoClip, VideoError>` | Automatically detects and decodes `.mp4` (unencrypted or `CENC`-encrypted) or raw `FVID` streams. |
| `CdmSession` / `EmeKeySystem` | W3C Encrypted Media Extensions (`"org.w3.clearkey"`, OS Media Foundation DRM, or External Plugin CDM) license session. |
| `inspect_mp4_metadata(&[u8]) -> Result<Mp4Metadata, VideoError>` | Inspects ISOBMFF (`ftyp`, `moov`, `trak`, `sinf`, `tenc`, `pssh`) headers and bounds before decoding. |
| `encode_encrypted_mp4(&VideoClip, key_id, key, iv) -> Result<Vec<u8>, VideoError>` | Encodes a `VideoClip` into an ISO/IEC 23001-7 `cenc` encrypted `.mp4` container. |
| `generate_demo_mp4` / `generate_encrypted_demo_mp4` | Generates synthetic unencrypted or `CENC`-encrypted `.mp4` test clips. |
| `validate_video_dimensions(width, height, frames) -> Result<(), VideoError>` | Verifies 4K resolution (`4096x2160`) and frame-count (`36,000`) security bounds. |

---

## Building and Testing

Ensure [`flux-compress`](https://github.com/buntatoes/flux-compress) (`../flux-compress`) and [`flux-gpu`](https://github.com/buntatoes/flux-gpu) (`../flux-gpu`) are cloned in sibling directories, then run:

```powershell
cargo test
```

---

## Security Policy

See [`SECURITY.md`](./SECURITY.md) for supported versions, 4K/container security ceilings, and vulnerability reporting instructions.

## License

Licensed under the Mozilla Public License Version 2.0 ([`MPL-2.0`](./LICENSE)).

## Author

**Buntos** ([@buntatoes](https://github.com/buntatoes))
