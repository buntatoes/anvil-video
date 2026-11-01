<div align="center">
  <h1>Anvil Video</h1>
  <p>CPU video (AVID NV12, ADCT 8x8 integer DCT/IDCT + motion compensation, RFC 6386 VP8 intra/inter bitstreams, and VP9/AV1 framing) and audio (PCM, IMA-ADPCM, Rice-coded FLAC frames, MDCT frames, and RFC 6716 Opus SILK/CELT packets) in small MP4, fMP4, and WebM containers.</p>
  <p>
    <img src="https://img.shields.io/badge/release-v.10-e8c37a?style=flat-square" alt="v.10" />
    <img src="https://img.shields.io/badge/status-Proof_of_Concept-d4a359?style=flat-square" alt="Proof of Concept" />
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Noncommercial_%28based_on_PolyForm_1.0.0%29_%7C_Paid_Commercial_%28Non--Compete%29-blue?style=flat-square" alt="Noncommercial (based on PolyForm 1.0.0) | Paid Commercial (Non-Compete)" /></a>
    <a href="./SECURITY.md"><img src="https://img.shields.io/badge/security-policy-2ea44f?style=flat-square" alt="Security" /></a>
    <img src="https://img.shields.io/badge/rust-2024-orange?style=flat-square" alt="Rust 2024" />
  </p>
</div>

Version 0.10.0 (`v.10`).

It needs these directories next to this one:

- [anvil-cipher](https://github.com/buntatoes/anvil-cipher)
- [anvil-compress](https://github.com/buntatoes/anvil-compress)
- [anvil-gpu](https://github.com/buntatoes/anvil-gpu)

## What it does

- **Video Codecs**: Encodes and decodes temporal `NV12` `AVID` streams (`GpuVideoSession`), 8x8 Integer DCT/IDCT + quantization + 16x16 macroblock motion-compensated `ADCT` `I-frame`/`P-frame` bitstreams (`encode_dct_mc_bitstream` / `decode_dct_mc_bitstream`), RFC 6386 `VP8` keyframe and interframe bitstreams (`src/vp8.rs`: `Vp8BoolEncoder` / `Vp8BoolDecoder` 8-bit boolean arithmetic coder, Partition 0 header parser, 16x16 luma / 8x8 chroma `DC_PRED`/`V_PRED`/`H_PRED`/`TM_PRED` and 4x4 `B_PRED` intra prediction, `ZEROMV` inter prediction, RFC 6386 Section 13.2 coefficient token trees, 4x4 Walsh-Hadamard and DCT/IDCT transforms `vp8_forward_wht_4x4` / `vp8_inverse_wht_4x4` / `vp8_forward_dct_4x4` / `vp8_inverse_idct_4x4`, and normal/simple macroblock loop filter `vp8_loop_filter_edge`, via `encode_vp8_rfc6386_stream` / `decode_vp8_rfc6386_stream`), clean-room `VP9` (`vp09`, Profile 0) bitstreams (`src/vp9.rs`: `Vp9BoolEncoder` / `Vp9BoolDecoder`, `[0x49, 0x83, 0x42]` sync code header `Vp9FrameHeader`, Annex-B superframe index packer/parser `build_vp9_superframe` / `parse_vp9_superframe_index`, 8x8 `DC_PRED`/`V_PRED`/`H_PRED`/`TM_PRED`/`D45_PRED`/`D135_PRED` intra and `ZEROMV`/`NEARESTMV`/`NEWMV` inter prediction across 3 reference slots, 4x4 ADST/IADST `vp9_forward_adst_4x4` / `vp9_inverse_iadst_4x4` and 8x8 DCT/IDCT transforms, via `encode_vp9_stream` / `decode_vp9_stream`), and clean-room `AV1` (`av01`, Main Profile 0) OBU bitstreams (`src/av1.rs`: `write_obu_leb128` / `read_obu_leb128` `OBU_TEMPORAL_DELIMITER`/`OBU_SEQUENCE_HEADER`/`OBU_FRAME` framing, `Av1SymbolEncoder` / `Av1SymbolDecoder` 15-bit CDF coder, 8x8 directional/smooth/Paeth intra and motion-compensated inter prediction, 8-direction CDEF search `av1_cdef_find_direction_8x8` + `av1_cdef_filter_4x4`, via `encode_av1_obu_stream` / `decode_av1_obu_stream`), while retaining backward compatibility with legacy `ADCT`-framed packets and Annex-B NAL unit framing (`encode_h264_annexb` / `decode_h264_annexb`).
- **Audio Codecs**: Encodes and decodes 16-bit `PCM` and 4-bit `IMA-ADPCM` (`encode_aaud_bitstream` / `decode_aaud_bitstream`), lossless linear-prediction + Rice-coded `fLaC` frames (`encode_flac_stream` / `decode_flac_stream`), 50%-overlapped `MDCT` audio frames (`encode_mdct_audio_stream` / `decode_mdct_audio_stream`), and RFC 6716 `Opus` packets (`src/opus.rs`: TOC byte and multi-frame packet splitter/packer for `Code 0..=3` CBR/VBR frames `split_opus_packet_frames` / `pack_opus_packet_frames`, `OpusRangeEncoder` / `OpusRangeDecoder` range coder, SILK LPC speech synthesis/analysis `encode_silk_frame` / `decode_silk_frame`, and 21-band CELT MDCT transform synthesis/analysis `encode_celt_frame` / `decode_celt_frame`, via `encode_rfc6716_opus_packet` / `decode_rfc6716_opus_packet`, while retaining backward compatibility with legacy `MDCT`-framed packets), plus RIFF `WAVE` output (`encode_pcm16_wav`) and `AudioOutputBridge` (with master audio clock `audio_clock_ms`, A/V sync drift calculation `av_drift_ms`, pure-Rust streaming PCM16 ringbuffer `drain_buffered_pcm`, and Windows `winmm.dll` `waveOutOpen` format query + `sndPlaySoundA` output).
- **Containers & Encryption**: ISOBMFF MP4 (`wrap_mp4_container`, `wrap_av_mp4_container`, `inspect_mp4_metadata`, `parse_mp4_container`), Fragmented MP4 (`fMP4` `moof`/`traf`/`trun` via `wrap_fmp4_init_segment`, `wrap_fmp4_media_segment`, `inspect_fmp4_segment`), `WebM` / Matroska EBML (`wrap_webm_container`, `inspect_webm_metadata`, `parse_webm_container`), and AES-128-CTR CENC sample encryption (`cenc_encrypt_sample`, `cenc_decrypt_sample`, `CdmSession`).
- **W3C Media Helpers & Probes**: `WebCodecsVideoEncoder` / `WebCodecsVideoDecoder` (backed by RFC 6386 `VP8`, `VP9`, `AV1`, and `ADCT`), `MseSourceBufferEngine` (accepting `fMP4` and `WebM` segments), `MediaCapabilitiesInfo`, and `probe_hardware_video_capabilities` (which checks for Windows `mfplat.dll` / `d3d11.dll` or Linux `/dev/dri/renderD128`).

## What it does not do

- It does not decode real H.264 (`avc1`) or third-party FLAC bitstreams from external sites. While `src/vp8.rs`, `src/vp9.rs`, `src/av1.rs`, and `src/opus.rs` implement clean-room boolean/CDF/range-coded intra/inter video frames and SILK/CELT audio frames, they do not implement every optional codec tool used by arbitrary third-party encoders (such as multi-tile CDF context adaptation, sub-pixel 8-tap Wiener/restoration filters, or commercial H.264 CABAC/CAVLC entropy coding), and the Annex-B H.264 helper wraps `ADCT` payloads inside its NAL headers.
- `probe_platform_gpu_backend` and `probe_hardware_video_capabilities` only inspect the host OS and presence of system libraries or device nodes. This crate does not open VA-API, DXVA2, or D3D11 Video hardware video decoders; all video decoding runs on the CPU before `anvil-gpu` blits `NV12` / `I420` / `RGBA` frames.

## Limits

- Width at most 4096, height at most 2160, and at most 8,847,360 pixels.
- Container size, packet size, box count, and box depth are capped.

## Build

```
cargo test
```

Rust 2024 edition, Rust 1.99.1 or newer, and Go 1.22 or newer (for the pure-Go `anvil-compress` 0.4.0 / `v.04` companion repository built by `build.rs`). Dependencies are the three sibling repositories above. No crates.io dependencies.

## The other repositories

The browser is [anvil-browser](https://github.com/buntatoes/anvil-browser) and the page engine is [anvil-engine](https://github.com/buntatoes/anvil-engine), built alongside the nine companion libraries below. None of them come from crates.io.

| Repository | What it is |
| --- | --- |
| [anvil-cipher](https://github.com/buntatoes/anvil-cipher) | AES-128, AES-256, CTR, AES-256-GCM, SHA-256, HMAC-SHA256, HKDF-SHA256 |
| [anvil-compress](https://github.com/buntatoes/anvil-compress) | DEFLATE, zlib, gzip |
| [anvil-hpack](https://github.com/buntatoes/anvil-hpack) | HPACK header compression for HTTP/2 |
| [anvil-image](https://github.com/buntatoes/anvil-image) | PNG, JPEG, GIF, WebP (`VP8L` + `VP8 `), ICO, BMP, PPM |
| [anvil-json](https://github.com/buntatoes/anvil-json) | JSON |
| [anvil-pqc](https://github.com/buntatoes/anvil-pqc) | ML-KEM-768, X25519, X25519MLKEM768, SHA-3 |
| [anvil-ttf](https://github.com/buntatoes/anvil-ttf) | TrueType and OpenType (`glyf`, `CFF `, `kern`, `GPOS`) |
| [anvil-gpu](https://github.com/buntatoes/anvil-gpu) | Compositor passes for boxes, shadows, gradients, and NV12/I420, on the CPU or a GPU |
| [anvil-video](https://github.com/buntatoes/anvil-video) | CPU frames (AVID/ADCT) and PCM/ADPCM/FLAC/MDCT audio in MP4, fMP4, and WebM containers |

## License

Dual-licensed under **noncommercial terms based on the PolyForm Noncommercial License 1.0.0** and a **Paid Commercial License with a Non-Compete Clause**. Copyright (c) 2026 Buntato. See [LICENSE](./LICENSE).
