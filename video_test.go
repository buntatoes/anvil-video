// SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0-OR-Commercial
// Copyright (c) 2026 Buntato. Verify canonical repository history at https://github.com/buntatoes

package anvilvideo

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

func makeTestPatternFrames(w, h, fps uint32, count int) []VideoFrame {
	out := make([]VideoFrame, count)
	stepMs := uint32(1000) / maxU32(fps, 1)
	for f := 0; f < count; f++ {
		rgba := make([]byte, int(w*h*4))
		for y := uint32(0); y < h; y++ {
			for x := uint32(0); x < w; x++ {
				idx := int((y*w + x) * 4)
				rgba[idx] = uint8((int(x)*11 + f*9) & 0xFF)
				rgba[idx+1] = uint8((int(y)*13 + f*5) & 0xFF)
				rgba[idx+2] = uint8(((int(x)+int(y))*7 + f*3) & 0xFF)
				rgba[idx+3] = 255
			}
		}
		frame, err := NewVideoFrame(w, h, uint32(f)*stepMs, f == 0, rgba)
		if err != nil {
			panic(err)
		}
		out[f] = frame
	}
	return out
}

func TestVersionAndValidation(t *testing.T) {
	if Version != "0.11.0" {
		t.Fatalf("expected Version 0.11.0, got %s", Version)
	}
	if _, err := NewVideoFrame(0, 16, 0, true, nil); err == nil {
		t.Fatal("expected error for zero width")
	}
	if _, err := NewVideoFrame(MaxVideoWidth+2, 16, 0, true, nil); err == nil {
		t.Fatal("expected error for width > MaxVideoWidth")
	}
}

func TestGpuVideoSessionEncodeDecodeAndBlit(t *testing.T) {
	session := DefaultGpuVideoSession()
	frames := makeTestPatternFrames(32, 32, 30, 3)
	bitstream, err := session.EncodeAvid(30, frames)
	if err != nil {
		t.Fatal(err)
	}
	clip, err := session.DecodeVideoAuto(bitstream)
	if err != nil {
		t.Fatal(err)
	}
	if len(clip.Frames) != 3 {
		t.Fatalf("expected 3 frames, got %d", len(clip.Frames))
	}
	target := make([]byte, 64*64*4)
	if err := clip.Frames[0].BlitInto(target, 64, 64, 64*4, 8, 8, 32, 32, true); err != nil {
		t.Fatal(err)
	}
	nonZero := false
	for _, b := range target {
		if b != 0 {
			nonZero = true
			break
		}
	}
	if !nonZero {
		t.Fatal("expected non-zero pixels after BlitInto")
	}
}

func TestDctMcAndAnnexBRoundtrip(t *testing.T) {
	frames := makeTestPatternFrames(32, 32, 24, 3)
	stream, err := EncodeDctMcBitstream(ProfileH264High, 24, 16, frames)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeDctMcBitstream(stream, false)
	if err != nil {
		t.Fatal(err)
	}
	if decoded.Fps != 24 || len(decoded.Frames) != 3 {
		t.Fatalf("unexpected decoded count=%d fps=%d", len(decoded.Frames), decoded.Fps)
	}

	annexB, err := EncodeH264AnnexB(24, 16, frames)
	if err != nil {
		t.Fatal(err)
	}
	decAnnexB, err := DecodeH264AnnexB(annexB, false)
	if err != nil {
		t.Fatal(err)
	}
	if decAnnexB.Fps != 24 || len(decAnnexB.Frames) != 3 {
		t.Fatalf("unexpected annex-b count=%d fps=%d", len(decAnnexB.Frames), decAnnexB.Fps)
	}
}

func TestVp8Rfc6386StreamRoundtrip(t *testing.T) {
	frames := makeTestPatternFrames(32, 32, 30, 2)
	stream, err := EncodeVp8Rfc6386Stream(30, 16, frames)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeVp8Rfc6386Stream(stream, false)
	if err != nil {
		t.Fatal(err)
	}
	if decoded.Fps != 30 || len(decoded.Frames) != 2 {
		t.Fatalf("unexpected vp8 count=%d fps=%d", len(decoded.Frames), decoded.Fps)
	}
	var block [16]int16
	for i := range block {
		block[i] = int16((i*7 - 20) * 3)
	}
	coeff := Vp8ForwardDct4x4(&block)
	recon := Vp8InverseIdct4x4(&coeff)
	for i := range block {
		if math.Abs(float64(block[i]-recon[i])) > 8 {
			t.Fatalf("4x4 DCT/IDCT drift at %d: %d vs %d", i, block[i], recon[i])
		}
	}
}

func TestVp9StreamAndSuperframeRoundtrip(t *testing.T) {
	frames := makeTestPatternFrames(32, 32, 30, 2)
	stream, err := EncodeVp9Stream(30, 16, frames)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeVp9Stream(stream, false)
	if err != nil {
		t.Fatal(err)
	}
	if decoded.Fps != 30 || len(decoded.Frames) != 2 {
		t.Fatalf("unexpected vp9 count=%d fps=%d", len(decoded.Frames), decoded.Fps)
	}

	sf := BuildVp9Superframe([][]byte{{1, 2, 3, 4}, {5, 6, 7}})
	sizes, ok := ParseVp9SuperframeIndex(sf)
	if !ok || len(sizes) != 2 || sizes[0] != 4 || sizes[1] != 3 {
		t.Fatalf("unexpected superframe index: ok=%v sizes=%v", ok, sizes)
	}
}

func TestAv1ObuStreamAndLeb128Roundtrip(t *testing.T) {
	var lebBuf bytes.Buffer
	WriteObuLeb128(&lebBuf, 624485)
	val, n, err := ReadObuLeb128(lebBuf.Bytes())
	if err != nil || val != 624485 || n != lebBuf.Len() {
		t.Fatalf("leb128 roundtrip failed: val=%d n=%d err=%v", val, n, err)
	}

	frames := makeTestPatternFrames(32, 32, 25, 2)
	stream, err := EncodeAv1ObuStream(25, 16, frames)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeAv1ObuStream(stream, false)
	if err != nil {
		t.Fatal(err)
	}
	if decoded.Fps != 25 || len(decoded.Frames) != 2 {
		t.Fatalf("unexpected av1 count=%d fps=%d", len(decoded.Frames), decoded.Fps)
	}
}

func TestAudioCodecsPcmAdpcmFlacMdctOpus(t *testing.T) {
	samples := make([]int16, 1920)
	for i := range samples {
		samples[i] = int16(math.Sin(float64(i)*0.11) * 12000.0)
	}
	pcmTrack := &AudioTrack{
		SampleRate: 48000,
		Channels:   2,
		Codec:      AudioPcm16Le,
		Samples:    samples,
	}

	// AAUD PCM & IMA-ADPCM
	aaudPcm, err := EncodeAaudBitstream(pcmTrack)
	if err != nil {
		t.Fatal(err)
	}
	decPcm, err := DecodeAaudBitstream(aaudPcm)
	if err != nil || decPcm.Codec != AudioPcm16Le || len(decPcm.Samples) != len(samples) {
		t.Fatalf("AAUD PCM roundtrip failed: err=%v codec=%v len=%d", err, decPcm.Codec, len(decPcm.Samples))
	}

	adpcmTrack := &AudioTrack{
		SampleRate: 48000,
		Channels:   2,
		Codec:      AudioImaAdpcm4,
		Samples:    samples,
	}
	aaudAdpcm, err := EncodeAaudBitstream(adpcmTrack)
	if err != nil {
		t.Fatal(err)
	}
	decAdpcm, err := DecodeAaudBitstream(aaudAdpcm)
	if err != nil || decAdpcm.Codec != AudioImaAdpcm4 || len(decAdpcm.Samples) != len(samples) {
		t.Fatalf("AAUD IMA-ADPCM roundtrip failed: err=%v codec=%v len=%d", err, decAdpcm.Codec, len(decAdpcm.Samples))
	}

	// Lossless FLAC
	flacBytes, err := EncodeFlacStream(pcmTrack)
	if err != nil {
		t.Fatal(err)
	}
	decFlac, err := DecodeFlacStream(flacBytes)
	if err != nil {
		t.Fatal(err)
	}
	if len(decFlac.Samples) != len(samples) {
		t.Fatalf("FLAC length mismatch: %d vs %d", len(decFlac.Samples), len(samples))
	}
	for i := range samples {
		if decFlac.Samples[i] != samples[i] {
			t.Fatalf("FLAC lossless mismatch at %d: %d vs %d", i, decFlac.Samples[i], samples[i])
		}
	}

	// MDCT Audio
	mdctBytes, err := EncodeMdctAudioStream(pcmTrack)
	if err != nil {
		t.Fatal(err)
	}
	decMdct, err := DecodeMdctAudioStream(mdctBytes)
	if err != nil || len(decMdct.Samples) != len(samples) {
		t.Fatalf("MDCT roundtrip failed: err=%v len=%d", err, len(decMdct.Samples))
	}

	// RFC 6716 Opus (CELT 48kHz & SILK 16kHz)
	opusCelt, err := EncodeRfc6716OpusPacket(pcmTrack)
	if err != nil {
		t.Fatal(err)
	}
	decOpusCelt, err := DecodeRfc6716OpusPacket(opusCelt)
	if err != nil || len(decOpusCelt.Samples) == 0 {
		t.Fatalf("Opus CELT roundtrip failed: err=%v", err)
	}

	silkTrack := &AudioTrack{
		SampleRate: 16000,
		Channels:   1,
		Codec:      AudioOpusMdct,
		Samples:    samples[:640],
	}
	opusSilk, err := EncodeRfc6716OpusPacket(silkTrack)
	if err != nil {
		t.Fatal(err)
	}
	decOpusSilk, err := DecodeRfc6716OpusPacket(opusSilk)
	if err != nil || len(decOpusSilk.Samples) == 0 {
		t.Fatalf("Opus SILK roundtrip failed: err=%v", err)
	}

	// WAV & AudioOutputBridge
	wavBytes := EncodePcm16Wav(48000, 2, samples)
	if !bytes.HasPrefix(wavBytes, []byte("RIFF")) {
		t.Fatal("expected RIFF header in WAV")
	}
	bridge := NewAudioOutputBridge()
	submitted := bridge.SubmitTrackWindow(pcmTrack, 0, 20, false)
	if submitted == 0 {
		t.Fatal("expected non-zero submitted samples")
	}
	drained := bridge.DrainBufferedPcm(480)
	if len(drained) == 0 {
		t.Fatal("expected non-empty drained samples")
	}
}

func TestContainersCencWebCodecsAndMse(t *testing.T) {
	sess := DefaultGpuVideoSession()
	frames := makeTestPatternFrames(16, 16, 24, 2)
	avidStream, err := sess.EncodeAvid(24, frames)
	if err != nil {
		t.Fatal(err)
	}
	audioTrack := SynthesizeHarmonicChime(100, 44100, 2, AudioImaAdpcm4)
	aaudStream, err := EncodeAaudBitstream(&audioTrack)
	if err != nil {
		t.Fatal(err)
	}

	// MP4 Audio+Video
	mp4Bytes := WrapAvMp4Container(16, 16, 100, avidStream, aaudStream)
	meta, err := InspectMp4Metadata(mp4Bytes)
	if err != nil {
		t.Fatal(err)
	}
	if !meta.HasAudioTrack || meta.Width != 16 || meta.Height != 16 {
		t.Fatalf("unexpected MP4 metadata: %+v", meta)
	}

	// Encrypted MP4 + EME ClearKey CDM
	encMp4, err := GenerateEncryptedDemoMp4(16, 16, 2)
	if err != nil {
		t.Fatal(err)
	}
	encMeta, err := InspectMp4Metadata(encMp4)
	if err != nil || !encMeta.Encrypted {
		t.Fatalf("expected encrypted MP4 metadata: %+v err=%v", encMeta, err)
	}
	decEncClip, err := sess.DecodeVideoAuto(encMp4)
	if err != nil || len(decEncClip.Frames) != 2 {
		t.Fatalf("DecodeVideoAuto on encrypted MP4 failed: err=%v frames=%d", err, len(decEncClip.Frames))
	}

	// WebM
	vStream, err := EncodeDctMcBitstream(ProfileH264High, 24, 16, frames)
	if err != nil {
		t.Fatal(err)
	}
	webmBytes := WrapWebmContainer(16, 16, 100, "V_VP9", vStream)
	webmMeta, err := InspectWebmMetadata(webmBytes)
	if err != nil || webmMeta.Width != 16 || webmMeta.VideoCodecID != "V_VP9" {
		t.Fatalf("unexpected WebM metadata: %+v err=%v", webmMeta, err)
	}

	// fMP4 & MSE
	initSeg := WrapFmp4InitSegment(16, 16, ProfileH264High, false)
	mediaSeg := WrapFmp4MediaSegment(1, 0, []uint32{40, 40}, vStream)
	mse := NewMseSourceBufferEngine("video/mp4")
	if _, err := mse.AppendBuffer(initSeg); err != nil {
		t.Fatal(err)
	}
	added, err := mse.AppendBuffer(mediaSeg)
	if err != nil || added != 2 || len(mse.Frames()) != 2 {
		t.Fatalf("MSE decode failed: err=%v added=%d count=%d", err, added, len(mse.Frames()))
	}

	// WebCodecs
	for _, codecStr := range []string{"vp8", "vp09.00.10.08", "av01.0.04M.08", "avc1.42E01E"} {
		wcEnc := NewWebCodecsVideoEncoder()
		if err := wcEnc.Configure(codecStr, 16, 16, 24, 16); err != nil {
			t.Fatal(err)
		}
		chunk, err := wcEnc.Encode(&frames[0], true)
		if err != nil {
			t.Fatal(err)
		}
		wcDec := NewWebCodecsVideoDecoder()
		if err := wcDec.Configure(codecStr); err != nil {
			t.Fatal(err)
		}
		outFrame, err := wcDec.Decode(&chunk)
		if err != nil || outFrame.Width != 16 || outFrame.Height != 16 {
			t.Fatalf("WebCodecs %s roundtrip failed: err=%v frame=%dx%d", codecStr, err, outFrame.Width, outFrame.Height)
		}
	}

	caps := QueryMediaCapabilities("video/mp4; codecs=\"av01.0.04M.08\"", 1920, 1080, 60)
	if !caps.Supported || !caps.Smooth {
		t.Fatalf("expected 1080p60 AV1 to be supported and smooth: %+v", caps)
	}
}

func TestSecurityBoundaryAndAdversarialBitstreams(t *testing.T) {
	// 1. MDCT packet with totalSamples > 0 and hopCount == 0 must be rejected without slice out-of-bounds panic.
	var badMdct [28]byte
	copy(badMdct[0:4], "MDCT")
	badMdct[4] = 1
	binary.BigEndian.PutUint32(badMdct[8:12], 48000)
	binary.BigEndian.PutUint32(badMdct[12:16], 100)
	binary.BigEndian.PutUint32(badMdct[16:20], 0)
	if _, err := DecodeMdctAudioStream(badMdct[:]); err == nil {
		t.Fatal("expected error on MDCT packet with hopCount=0 and totalSamples=100")
	}

	// 2. VP9 subframe with overflowing part0Len / part1Len (0x7FFFFFFF + 0x7FFFFFFF) must not overflow int or panic.
	var badVp9 [28]byte
	badVp9[0] = 0x82
	badVp9[1] = 0x49
	badVp9[2] = 0x83
	badVp9[3] = 0x42
	binary.BigEndian.PutUint16(badVp9[4:6], 15)
	binary.BigEndian.PutUint16(badVp9[6:8], 15)
	copy(badVp9[8:12], "VP9B")
	binary.BigEndian.PutUint32(badVp9[12:16], 0x7FFFFFFF)
	binary.BigEndian.PutUint32(badVp9[16:20], 0x7FFFFFFF)
	if _, err := DecodeVp9Bitstream(badVp9[:], false); err == nil {
		t.Fatal("expected error on VP9 subframe with overflowing partition lengths")
	}

	// 3. VP8 bitstream with forged extraCount = 0xFFFFFFFF in VP8F trailer must fail with ErrFrameCountTooLarge.
	frames := makeTestPatternFrames(16, 16, 24, 1)
	vp8Valid, err := EncodeVp8Rfc6386Bitstream(24, 16, frames)
	if err != nil {
		t.Fatal(err)
	}
	badVp8 := append([]byte(nil), vp8Valid...)
	if pos := bytes.Index(badVp8, []byte("VP8F")); pos >= 0 && pos+12 <= len(badVp8) {
		binary.LittleEndian.PutUint32(badVp8[pos+8:pos+12], 0xFFFFFFFF)
	}
	if _, err := DecodeVp8Rfc6386Bitstream(badVp8, false); err == nil {
		t.Fatal("expected error on VP8 bitstream with extraCount=0xFFFFFFFF")
	}

	// 4. Forged trun sample count in fMP4 must be clamped to actual payload entries.
	mediaSeg := WrapFmp4MediaSegment(1, 0, []uint32{40, 40}, []byte{1, 2, 3, 4})
	if idx := bytes.Index(mediaSeg, []byte("trun")); idx >= 0 && idx+12 <= len(mediaSeg) {
		binary.BigEndian.PutUint32(mediaSeg[idx+8:idx+12], 0x7FFFFFFF)
	}
	info, err := InspectFmp4Segment(mediaSeg)
	if err != nil {
		t.Fatalf("InspectFmp4Segment failed: %v", err)
	}
	if info.SampleCount > 16 {
		t.Fatalf("expected forged trun SampleCount to be bounded by payload size, got %d", info.SampleCount)
	}

	// 5. Truncated NV12/I420 surfaces, negative DrainBufferedPcm, NaN clamp, and MaxInt32 BlitInto must not panic.
	if got := clampF64(math.NaN(), -1.0, 1.0); got != -1.0 {
		t.Fatalf("clampF64(NaN) = %v, want -1.0", got)
	}
	truncNv12 := Nv12Surface{Width: 16, Height: 16, YPlane: make([]byte, 4), UVPlane: make([]byte, 2)}
	if truncNv12.ToRgbaBt709() != nil {
		t.Fatal("expected nil from truncated Nv12Surface.ToRgbaBt709")
	}
	truncI420 := I420Surface{Width: 16, Height: 16, YPlane: make([]byte, 4), UPlane: make([]byte, 2), VPlane: make([]byte, 2)}
	if truncI420.ToRgbaBt709() != nil {
		t.Fatal("expected nil from truncated I420Surface.ToRgbaBt709")
	}
	bridge := NewAudioOutputBridge()
	if bridge.DrainBufferedPcm(-5) != nil {
		t.Fatal("expected nil from negative DrainBufferedPcm")
	}
	dst := make([]byte, 16*16*4)
	if err := frames[0].BlitInto(dst, 16, 16, 16*4, math.MaxInt32, math.MaxInt32, 8, 8, false); err != nil {
		t.Fatalf("BlitInto at MaxInt32 returned unexpected error: %v", err)
	}
}
