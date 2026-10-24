<div align="center">
  <h1>root-video</h1>
  <p>CPU frames in a small MP4 container, plus AES-128-CTR samples.</p>
  <p>
    <img src="https://img.shields.io/badge/release-v.04-e8c37a?style=flat-square" alt="v.04" />
    <img src="https://img.shields.io/badge/status-Proof_of_Concept-d4a359?style=flat-square" alt="Proof of Concept" />
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MPL--2.0-blue?style=flat-square" alt="MPL-2.0" /></a>
    <a href="./SECURITY.md"><img src="https://img.shields.io/badge/security-policy-2ea44f?style=flat-square" alt="Security" /></a>
    <img src="https://img.shields.io/badge/rust-2024-orange?style=flat-square" alt="Rust 2024" />
  </p>
</div>

Version 0.4.0.

It needs these directories next to this one:

- [root-cipher](https://github.com/buntatoes/root-cipher)
- [root-compress](https://github.com/buntatoes/root-compress)
- [root-gpu](https://github.com/buntatoes/root-gpu)

## What it does

- Frames are RGBA on the CPU. `generate_demo_mp4` builds a short clip. `GpuVideoSession` encodes and decodes that clip.
- The container is an ISOBMFF / MP4 shell (`ftyp`, `moov`, `mdat`). `wrap_mp4_container`, `inspect_mp4_metadata`, and `parse_mp4_container` read and write it.
- `cenc_encrypt_sample` and `cenc_decrypt_sample` are AES-128-CTR from `root-cipher`. The IV is the initial 16-byte counter.
- `CdmSession` holds keys in memory for the demo clip. It is not a CDM and it does not talk to a license server.
- Frame bytes are compressed with `root-compress`. Color conversion uses `Nv12Surface` from `root-gpu`, still on the CPU.

## What it does not do

- It does not decode H.264, AV1, or any other production codec. Those names exist as labels on `VideoCodecProfile`.
- `probe_platform_gpu_backend` returns a name for the operating system. It does not open Direct3D, VA-API, or a GPU.

## Limits

- Width at most 4096, height at most 2160, and at most 8,847,360 pixels.
- Container size, packet size, box count, and box depth are capped.

## Build

```
cargo test
```

Rust 2024. Dependencies are the three path crates above. No crates.io dependencies.

## The other repositories

The browser is [root-browser](https://github.com/buntatoes/root-browser). These libraries sit next to it. None of them come from crates.io.

| Repository | What it is |
| --- | --- |
| [root-cipher](https://github.com/buntatoes/root-cipher) | AES-128, AES-256, CTR, AES-256-GCM |
| [root-compress](https://github.com/buntatoes/root-compress) | DEFLATE, zlib, gzip |
| [root-image](https://github.com/buntatoes/root-image) | PNG, JPEG, GIF, WebP, ICO, BMP, PPM |
| [root-json](https://github.com/buntatoes/root-json) | JSON |
| [root-ttf](https://github.com/buntatoes/root-ttf) | TrueType and OpenType |
| [root-gpu](https://github.com/buntatoes/root-gpu) | CPU drawing of boxes, shadows, and NV12 |
| [root-video](https://github.com/buntatoes/root-video) | CPU frames in a small MP4 container |

## License

MPL-2.0. Copyright (c) 2026 Buntos.
