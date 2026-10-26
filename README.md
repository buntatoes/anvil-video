<div align="center">
  <h1>anvil-video</h1>
  <p>CPU frames in a small MP4 container, plus AES-128-CTR samples.</p>
  <p>
    <img src="https://img.shields.io/badge/release-v.07-e8c37a?style=flat-square" alt="v.07" />
    <img src="https://img.shields.io/badge/status-Proof_of_Concept-d4a359?style=flat-square" alt="Proof of Concept" />
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MPL--2.0-blue?style=flat-square" alt="MPL-2.0" /></a>
    <a href="./SECURITY.md"><img src="https://img.shields.io/badge/security-policy-2ea44f?style=flat-square" alt="Security" /></a>
    <img src="https://img.shields.io/badge/rust-2024-orange?style=flat-square" alt="Rust 2024" />
  </p>
</div>

Version 0.7.0.

It needs these directories next to this one:

- [anvil-cipher](https://github.com/buntatoes/anvil-cipher)
- [anvil-compress](https://github.com/buntatoes/anvil-compress)
- [anvil-gpu](https://github.com/buntatoes/anvil-gpu)

## What it does

- Frames are RGBA on the CPU. `generate_demo_mp4` builds a short clip. `GpuVideoSession` encodes and decodes that clip.
- The container is a small MP4 shell (`ftyp`, `moov`, `mdat`). `wrap_mp4_container`, `wrap_av_mp4_container`, `inspect_mp4_metadata`, and `parse_mp4_container` read and write video-only (`AVID`) and video+audio (`AAVM`, with 16-bit PCM or 4-bit IMA-ADPCM `AAUD` audio) files. `encode_pcm16_wav` writes a `.wav` buffer, and `AudioOutputBridge` tracks audio levels and can hand that buffer to `winmm.dll` on Windows.
- `cenc_encrypt_sample` and `cenc_decrypt_sample` are AES-128-CTR from `anvil-cipher`. The IV is the initial 16-byte counter.
- `CdmSession` holds keys in memory for the demo clip. It is not a CDM and it does not talk to a license server.
- Frame bytes are compressed with `anvil-compress`. When the hardware flag is on, color conversion and present go through `anvil-gpu`. The `AVID` packets are still zlib. This is not an H.264 or AV1 decoder. The browser draws the decoded frame through `composite_rgba_frame` before it paints the controls.

## What it does not do

- It does not decode H.264, AV1, or any other production codec. Those names exist as labels on `VideoCodecProfile`.
- `probe_platform_gpu_backend` still returns a name for the operating system. Plane conversion runs on the anvil-gpu software device. This crate does not open VA-API or Direct3D.

## Limits

- Width at most 4096, height at most 2160, and at most 8,847,360 pixels.
- Container size, packet size, box count, and box depth are capped.

## Build

```
cargo test
```

Rust 2024 edition, Rust 1.85.1 or newer. Dependencies are the three path crates above. No crates.io dependencies.

## The other repositories

The browser is [anvil-browser](https://github.com/buntatoes/anvil-browser) (Apache-2.0). The page engine is [anvil-engine](https://github.com/buntatoes/anvil-engine). Its source is public to read and audit, but it's proprietary, not MPL-2.0, and can't be redistributed on its own. These nine libraries sit next to both, and all of them are MPL-2.0. None of them come from crates.io.

| Repository | What it is |
| --- | --- |
| [anvil-cipher](https://github.com/buntatoes/anvil-cipher) | AES-128, AES-256, CTR, AES-256-GCM |
| [anvil-compress](https://github.com/buntatoes/anvil-compress) | DEFLATE, zlib, gzip |
| [anvil-hpack](https://github.com/buntatoes/anvil-hpack) | HPACK header compression for HTTP/2 |
| [anvil-image](https://github.com/buntatoes/anvil-image) | PNG, JPEG, GIF, WebP, ICO, BMP, PPM |
| [anvil-json](https://github.com/buntatoes/anvil-json) | JSON |
| [anvil-pqc](https://github.com/buntatoes/anvil-pqc) | ML-KEM-768, X25519, X25519MLKEM768, SHA-3 (not used by the browser yet) |
| [anvil-ttf](https://github.com/buntatoes/anvil-ttf) | TrueType and OpenType (`glyf`, `CFF `, `kern`, `GPOS`) |
| [anvil-gpu](https://github.com/buntatoes/anvil-gpu) | Compositor passes for boxes, shadows, gradients, and NV12, on the CPU or a GPU |
| [anvil-video](https://github.com/buntatoes/anvil-video) | CPU frames and PCM/ADPCM audio in a small MP4 container |

## License

MPL-2.0. Copyright (c) 2026 Buntato.
