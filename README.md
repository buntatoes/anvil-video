<div align="center">
  <h1>Anvil Video</h1>
  <p>Pure-Go CPU video (AVID NV12, ADCT 8x8 integer DCT/IDCT + motion compensation, RFC 6386 VP8 intra/inter bitstreams, and VP9/AV1 framing) and audio (PCM, IMA-ADPCM, Rice-coded FLAC frames, MDCT frames, and RFC 6716 Opus SILK/CELT packets) in small MP4, fMP4, and WebM containers.</p>
  <p>
    <img src="https://img.shields.io/badge/release-v.11-e8c37a?style=flat-square" alt="v.11" />
    <img src="https://img.shields.io/badge/status-Proof_of_Concept-d4a359?style=flat-square" alt="Proof of Concept" />
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Noncommercial_%28based_on_PolyForm_1.0.0%29_%7C_Paid_Commercial_%28Non--Compete%29-blue?style=flat-square" alt="Noncommercial (based on PolyForm 1.0.0) | Paid Commercial (Non-Compete)" /></a>
    <a href="./SECURITY.md"><img src="https://img.shields.io/badge/security-policy-2ea44f?style=flat-square" alt="Security" /></a>
    <img src="https://img.shields.io/badge/go-1.22+-00ADD8?style=flat-square" alt="Go 1.22+" />
  </p>
</div>

Version 0.11.0 (`v.11`).

100% pure Go (`module github.com/buntatoes/anvil-video`) with zero external module dependencies.

## What it does

- **Video Codecs**: Encodes and decodes temporal `NV12` `AVID` streams (`GpuVideoSession`), 8x8 Integer DCT/IDCT + quantization + 16x16 macroblock motion-compensated `ADCT` `I-frame`/`P-frame` bitstreams (`EncodeDctMcBitstream` / `DecodeDctMcBitstream`), RFC 6386 `VP8` keyframe and interframe bitstreams (`vp8.go`: `Vp8BoolEncoder` / `Vp8BoolDecoder` 8-bit boolean arithmetic coder, Partition 0 header parser, 16x16 luma / 8x8 chroma `DC_PRED`/`V_PRED`/`H_PRED`/`TM_PRED` and 4x4 `B_PRED` intra prediction, `ZEROMV` inter prediction, RFC 6386 Section 13.2 coefficient token trees, 4x4 Walsh-Hadamard and DCT/IDCT transforms `Vp8ForwardWht4x4` / `Vp8InverseWht4x4` / `Vp8ForwardDct4x4` / `Vp8InverseIdct4x4`, and normal/simple macroblock loop filter `Vp8LoopFilterEdge`, via `EncodeVp8Rfc6386Stream` / `DecodeVp8Rfc6386Stream`), clean-room `VP9` (`vp09`, Profile 0) bitstreams (`vp9.go`: `Vp9BoolEncoder` / `Vp9BoolDecoder`, `[0x49, 0x83, 0x42]` sync code header `Vp9FrameHeader`, Annex-B superframe index packer/parser `BuildVp9Superframe` / `ParseVp9SuperframeIndex`, 8x8 `DC_PRED`/`V_PRED`/`H_PRED`/`TM_PRED`/`D45_PRED`/`D135_PRED` intra and `ZEROMV`/`NEARESTMV`/`NEWMV` inter prediction across 3 reference slots, 4x4 ADST/IADST `Vp9ForwardAdst4x4` / `Vp9InverseIadst4x4` and 8x8 DCT/IDCT transforms, via `EncodeVp9Stream` / `DecodeVp9Stream`), and clean-room `AV1` (`av01`, Main Profile 0) OBU bitstreams (`av1.go`: `WriteObuLeb128` / `ReadObuLeb128` `ObuTemporalDelimiter`/`ObuSequenceHeader`/`ObuFrame` framing, `Av1SymbolEncoder` / `Av1SymbolDecoder` 15-bit CDF coder, 8x8 directional/smooth/Paeth intra and motion-compensated inter prediction, 8-direction CDEF search `Av1CdefFindDirection8x8` + `Av1CdefFilter4x4`, via `EncodeAv1ObuStream` / `DecodeAv1ObuStream`), while retaining backward compatibility with legacy `ADCT`-framed packets and Annex-B NAL unit framing (`EncodeH264AnnexB` / `DecodeH264AnnexB`).
- **Audio Codecs**: Encodes and decodes 16-bit `PCM` and 4-bit `IMA-ADPCM` (`EncodeAaudBitstream` / `DecodeAaudBitstream`), lossless linear-prediction + Rice-coded `fLaC` frames (`EncodeFlacStream` / `DecodeFlacStream`), 50%-overlapped `MDCT` audio frames (`EncodeMdctAudioStream` / `DecodeMdctAudioStream`), and RFC 6716 `Opus` packets (`opus.go`: TOC byte and multi-frame packet splitter/packer for `Code 0..=3` CBR/VBR frames `SplitOpusPacketFrames` / `PackOpusPacketFrames`, `OpusRangeEncoder` / `OpusRangeDecoder` range coder, SILK LPC speech synthesis/analysis `EncodeSilkFrame` / `DecodeSilkFrame`, and 21-band CELT MDCT transform synthesis/analysis `EncodeCeltFrame` / `DecodeCeltFrame`, via `EncodeRfc6716OpusPacket` / `DecodeRfc6716OpusPacket`, while retaining backward compatibility with legacy `MDCT`-framed packets), plus RIFF `WAVE` output (`EncodePcm16Wav`) and `AudioOutputBridge` (with master audio clock `AudioClockMs`, A/V sync drift calculation `AvDriftMs`, and streaming PCM16 ringbuffer `DrainBufferedPcm`).
- **Containers & Encryption**: ISOBMFF MP4 (`WrapMp4Container`, `WrapAvMp4Container`, `InspectMp4Metadata`, `ParseMp4Container`), Fragmented MP4 (`fMP4` `moof`/`traf`/`trun` via `WrapFmp4InitSegment`, `WrapFmp4MediaSegment`, `InspectFmp4Segment`), `WebM` / Matroska EBML (`WrapWebmContainer`, `InspectWebmMetadata`, `ParseWebmContainer`), and AES-128-CTR CENC sample encryption (`CencEncryptSample`, `CencDecryptSample`, `CdmSession`).
- **W3C Media Helpers & Probes**: `WebCodecsVideoEncoder` / `WebCodecsVideoDecoder` (backed by RFC 6386 `VP8`, `VP9`, `AV1`, and `ADCT`), `MseSourceBufferEngine` (accepting `fMP4` and `WebM` segments), `MediaCapabilitiesInfo`, and `ProbeHardwareVideoCapabilities` (which checks for Windows `mfplat.dll` / `d3d11.dll` or Linux `/dev/dri/renderD128`).

## What it does not do

- It does not decode real H.264 (`avc1`) or third-party FLAC bitstreams from external sites. While `vp8.go`, `vp9.go`, `av1.go`, and `opus.go` implement clean-room boolean/CDF/range-coded intra/inter video frames and SILK/CELT audio frames, they do not implement every optional codec tool used by arbitrary third-party encoders (such as multi-tile CDF context adaptation, sub-pixel 8-tap Wiener/restoration filters, or commercial H.264 CABAC/CAVLC entropy coding), and the Annex-B H.264 helper wraps `ADCT` payloads inside its NAL headers.
- `ProbePlatformGpuBackend` and `ProbeHardwareVideoCapabilities` only inspect the host OS and presence of system libraries or device nodes. This module does not open VA-API, DXVA2, or D3D11 Video hardware video decoders; all video decoding runs on the CPU before `anvil-gpu` blits `NV12` / `I420` / `RGBA` frames.

## Limits

- Width at most 4096, height at most 2160, and at most 8,847,360 pixels.
- Container size, packet size, box count, and box depth are capped.

## Build

```
go test -v ./...
```

Go 1.22 or newer (`go1.27.3` verified). Zero third-party module dependencies.

## The other repositories

The browser is [anvil-browser](https://github.com/buntatoes/anvil-browser) and the page engine is [anvil-engine](https://github.com/buntatoes/anvil-engine), built alongside the nine companion libraries below.

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
| [anvil-video](https://github.com/buntatoes/anvil-video) | CPU frames (AVID/ADCT/VP8/VP9/AV1) and PCM/ADPCM/FLAC/MDCT/Opus audio in MP4, fMP4, and WebM containers |

## License

Dual-licensed under **noncommercial terms based on the PolyForm Noncommercial License 1.0.0** and a **Paid Commercial License with a Non-Compete Clause**. Copyright (c) 2026 Buntato. See [LICENSE](./LICENSE).
