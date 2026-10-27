<div align="center">
  <h1>anvil-video</h1>
  <p>CPU video (AVID NV12 and ADCT 8x8 integer DCT/IDCT + motion compensation) and audio (PCM, IMA-ADPCM, Rice-coded FLAC frames, and MDCT frames) in small MP4, fMP4, and WebM containers.</p>
  <p>
    <img src="https://img.shields.io/badge/release-v.08-e8c37a?style=flat-square" alt="v.08" />
    <img src="https://img.shields.io/badge/status-Proof_of_Concept-d4a359?style=flat-square" alt="Proof of Concept" />
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-PolyForm_Noncommercial_1.0.0_%7C_Paid_Commercial_%28Non--Compete%29-blue?style=flat-square" alt="PolyForm Noncommercial 1.0.0 | Paid Commercial (Non-Compete)" /></a>
    <a href="./SECURITY.md"><img src="https://img.shields.io/badge/security-policy-2ea44f?style=flat-square" alt="Security" /></a>
    <img src="https://img.shields.io/badge/rust-2024-orange?style=flat-square" alt="Rust 2024" />
  </p>
</div>

Version 0.8.0 (`v.08`).

It needs these directories next to this one:

- [anvil-cipher](https://github.com/buntatoes/anvil-cipher)
- [anvil-compress](https://github.com/buntatoes/anvil-compress)
- [anvil-gpu](https://github.com/buntatoes/anvil-gpu)

## What it does

- **Video Codecs**: Encodes and decodes temporal `NV12` `AVID` streams (`GpuVideoSession`) and 8x8 Integer DCT/IDCT + quantization + 16x16 macroblock motion-compensated `ADCT` `I-frame`/`P-frame` bitstreams (`encode_dct_mc_bitstream` / `decode_dct_mc_bitstream`), plus Annex-B NAL unit framing (`encode_h264_annexb` / `decode_h264_annexb`) and OBU framing (`encode_av1_obu_stream` / `decode_av1_obu_stream`) around `ADCT` payloads.
- **Audio Codecs**: Encodes and decodes 16-bit `PCM` and 4-bit `IMA-ADPCM` (`encode_aaud_bitstream` / `decode_aaud_bitstream`), lossless linear-prediction + Rice-coded `fLaC` frames (`encode_flac_stream` / `decode_flac_stream`), and 50%-overlapped `MDCT` audio frames (`encode_mdct_audio_stream` / `decode_mdct_audio_stream`), plus RIFF `WAVE` output (`encode_pcm16_wav`) and `AudioOutputBridge`.
- **Containers & Encryption**: ISOBMFF MP4 (`wrap_mp4_container`, `wrap_av_mp4_container`, `inspect_mp4_metadata`, `parse_mp4_container`), Fragmented MP4 (`fMP4` `moof`/`traf`/`trun` via `wrap_fmp4_init_segment`, `wrap_fmp4_media_segment`, `inspect_fmp4_segment`), `WebM` / Matroska EBML (`wrap_webm_container`, `inspect_webm_metadata`, `parse_webm_container`), and AES-128-CTR CENC sample encryption (`cenc_encrypt_sample`, `cenc_decrypt_sample`, `CdmSession`).
- **W3C Media Helpers & Probes**: `WebCodecsVideoEncoder` / `WebCodecsVideoDecoder` (backed by `ADCT`), `MseSourceBufferEngine` (accepting `fMP4` and `WebM` segments), `MediaCapabilitiesInfo`, and `probe_hardware_video_capabilities` (which checks for Windows `mfplat.dll` / `d3d11.dll` or Linux `/dev/dri/renderD128`).

## What it does not do

- It does not decode real H.264 (`avc1`), AV1 (`av01`), VP9 (`vp09`), Opus, or third-party FLAC bitstreams from external sites. The Annex-B NAL and OBU helpers wrap and unwrap `ADCT` payloads produced by this crate, and the codec strings (`avc1.*`, `av01.*`, `vp09.*`) select those internal framing paths.
- `probe_platform_gpu_backend` and `probe_hardware_video_capabilities` only inspect the host OS and presence of system libraries or device nodes. This crate does not open VA-API, DXVA2, or D3D11 Video hardware video decoders; all video decoding runs on the CPU before `anvil-gpu` blits `NV12` / `I420` / `RGBA` frames.

## Limits

- Width at most 4096, height at most 2160, and at most 8,847,360 pixels.
- Container size, packet size, box count, and box depth are capped.

## Build

```
cargo test
```

Rust 2024 edition, Rust 1.99.1 or newer. Dependencies are the three path crates above. No crates.io dependencies.

## The other repositories

The browser is [anvil-browser](https://github.com/buntatoes/anvil-browser) and the page engine is [anvil-engine](https://github.com/buntatoes/anvil-engine), built alongside the eleven companion libraries below. None of them come from crates.io.

| Repository | What it is |
| --- | --- |
| [anvil-cipher](https://github.com/buntatoes/anvil-cipher) | AES-128, AES-256, CTR, AES-256-GCM, SHA-256, HMAC-SHA256, HKDF-SHA256 |
| [anvil-compress](https://github.com/buntatoes/anvil-compress) | DEFLATE, zlib, gzip |
| [anvil-css](https://github.com/buntatoes/anvil-css) | CSS Syntax Level 3 tokenizer, Selectors Level 4 matcher, and property/media/@supports parser |
| [anvil-hpack](https://github.com/buntatoes/anvil-hpack) | HPACK header compression for HTTP/2 |
| [anvil-image](https://github.com/buntatoes/anvil-image) | PNG, JPEG, GIF, WebP (`VP8L` + `VP8 `), ICO, BMP, PPM |
| [anvil-js](https://github.com/buntatoes/anvil-js) | Self-hosted ES2022+ JavaScript compiler and Core JS Web/DOM runtime for `AnvilVM` |
| [anvil-json](https://github.com/buntatoes/anvil-json) | JSON |
| [anvil-pqc](https://github.com/buntatoes/anvil-pqc) | ML-KEM-768, X25519, X25519MLKEM768, SHA-3 |
| [anvil-ttf](https://github.com/buntatoes/anvil-ttf) | TrueType and OpenType (`glyf`, `CFF `, `kern`, `GPOS`) |
| [anvil-gpu](https://github.com/buntatoes/anvil-gpu) | Compositor passes for boxes, shadows, gradients, and NV12/I420, on the CPU or a GPU |
| [anvil-video](https://github.com/buntatoes/anvil-video) | CPU frames (AVID/ADCT) and PCM/ADPCM/FLAC/MDCT audio in MP4, fMP4, and WebM containers |

## License

Dual-licensed under the **PolyForm Noncommercial License 1.0.0** and a **Paid Commercial License with a Non-Compete Clause**. Copyright (c) 2026 Buntato. See [LICENSE](./LICENSE).
