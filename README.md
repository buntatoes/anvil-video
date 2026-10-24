# flux-video

CPU video frames, a small MP4 container, and AES-128-CTR samples. Version 0.4.0.

It needs these directories next to this one:

- [flux-cipher](https://github.com/buntatoes/flux-cipher)
- [flux-compress](https://github.com/buntatoes/flux-compress)
- [flux-gpu](https://github.com/buntatoes/flux-gpu)

## What it does

- Frames are RGBA on the CPU. `generate_demo_mp4` builds a short clip. `GpuVideoSession` encodes and decodes that clip.
- The container is an ISOBMFF / MP4 shell (`ftyp`, `moov`, `mdat`). `wrap_mp4_container`, `inspect_mp4_metadata`, and `parse_mp4_container` read and write it.
- `cenc_encrypt_sample` and `cenc_decrypt_sample` are AES-128-CTR from `flux-cipher`. The IV is the initial 16-byte counter.
- `CdmSession` holds keys in memory for the demo clip. It is not a CDM and it does not talk to a license server.
- Frame bytes are compressed with `flux-compress`. Color conversion uses `Nv12Surface` from `flux-gpu`, still on the CPU.

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

The browser is [flux-browser](https://github.com/buntatoes/flux-browser). These libraries sit next to it. None of them come from crates.io.

| Repository | What it is |
| --- | --- |
| [flux-cipher](https://github.com/buntatoes/flux-cipher) | AES-128, AES-256, CTR, AES-256-GCM |
| [flux-compress](https://github.com/buntatoes/flux-compress) | DEFLATE, zlib, gzip |
| [flux-image](https://github.com/buntatoes/flux-image) | PNG, JPEG, GIF, WebP, ICO, BMP, PPM |
| [flux-json](https://github.com/buntatoes/flux-json) | JSON |
| [flux-ttf](https://github.com/buntatoes/flux-ttf) | TrueType and OpenType |
| [flux-gpu](https://github.com/buntatoes/flux-gpu) | CPU drawing of boxes, shadows, and NV12 |
| [flux-video](https://github.com/buntatoes/flux-video) | CPU frames in a small MP4 container |

## License

MPL-2.0. Copyright (c) 2026 Buntos.
