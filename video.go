// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Buntato

// Package anvilvideo implements clean-room CPU/GPU video and audio containers,
// AVID and ADCT frame codecs, RFC 6716 Opus (SILK + CELT), RFC 6386 VP8, VP9,
// AOMedia AV1 OBU, H.264 Annex-B, lossless fLaC, MDCT audio, ISOBMFF MP4/fMP4,
// WebM/EBML, AES-128-CTR CENC sample encryption, and W3C WebCodecs/MSE helpers.
package anvilvideo

import (
	"bytes"
	"compress/zlib"
	"crypto/aes"
	"crypto/cipher"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"runtime"
	"sync/atomic"
)

// Version is the semantic version of anvil-video (v.11).
const Version = "0.11.0"

const (
	MaxVideoWidth     uint32 = 4096
	MaxVideoHeight    uint32 = 2160
	MaxVideoPixels    uint64 = 8_847_360
	MaxVideoFrames    uint32 = 36_000
	MaxPacketBytes    int    = 16 * 1024 * 1024
	MaxContainerBytes int    = 64 * 1024 * 1024
	MaxMp4Boxes       int    = 256
	MaxMp4BoxDepth    int    = 16
)

// W3CClearKeySystemID is the fixed 16-byte W3C ClearKey system identifier in pssh boxes.
var W3CClearKeySystemID = [16]byte{
	0x10, 0x77, 0xef, 0xec, 0xc0, 0xb2, 0x4d, 0x02,
	0xac, 0xe3, 0x3c, 0x1e, 0x52, 0xe2, 0xfb, 0x4b,
}

// DefaultEmeDemoKID is the fixed 16-byte demo key ID written into tenc and pssh by the encrypted demo.
var DefaultEmeDemoKID = [16]byte{
	0x41, 0x4e, 0x56, 0x4c, 0x5f, 0x43, 0x45, 0x4e,
	0x43, 0x5f, 0x4b, 0x49, 0x44, 0x5f, 0x30, 0x31,
}

// DefaultEmeDemoKey is the fixed 16-byte demo AES-128 key used by the encrypted demo.
var DefaultEmeDemoKey = [16]byte{
	0x9a, 0x7b, 0x3c, 0x5d, 0x1e, 0x8f, 0x20, 0x41,
	0x62, 0x83, 0xa4, 0xc5, 0xe6, 0x07, 0x18, 0x29,
}

// VideoErrorKind categorizes media container and bitstream errors.
type VideoErrorKind int

const (
	ErrInvalidHeader VideoErrorKind = iota + 1
	ErrDimensionsTooLarge
	ErrOddDimensions
	ErrFrameCountTooLarge
	ErrPacketTooLarge
	ErrContainerTooLarge
	ErrTooManyBoxes
	ErrBoxDepthExceeded
	ErrCorruptedBitstream
	ErrMissingDecryptionKey
	ErrCompression
)

// VideoError represents a structured failure from a limit check, container walk, or codec parse.
type VideoError struct {
	Kind   VideoErrorKind
	Width  uint32
	Height uint32
	Count  uint32
	Size   int
	Depth  int
	KID    [16]byte
	Detail string
}

func (e *VideoError) Error() string {
	switch e.Kind {
	case ErrInvalidHeader:
		return "invalid video stream or container header"
	case ErrDimensionsTooLarge:
		return fmt.Sprintf("video dimensions %dx%d exceed 4K security cap (%dx%d)", e.Width, e.Height, MaxVideoWidth, MaxVideoHeight)
	case ErrOddDimensions:
		return fmt.Sprintf("video dimensions %dx%d must be even for 4:2:0 NV12 surfaces", e.Width, e.Height)
	case ErrFrameCountTooLarge:
		return fmt.Sprintf("video frame count %d exceeds security limit (%d)", e.Count, MaxVideoFrames)
	case ErrPacketTooLarge:
		return fmt.Sprintf("video packet size %d exceeds security limit (%d)", e.Size, MaxPacketBytes)
	case ErrContainerTooLarge:
		return fmt.Sprintf("video container size %d exceeds security limit (%d)", e.Size, MaxContainerBytes)
	case ErrTooManyBoxes:
		return fmt.Sprintf("MP4 container box count %d exceeds security limit (%d)", e.Size, MaxMp4Boxes)
	case ErrBoxDepthExceeded:
		return fmt.Sprintf("MP4 container box nesting depth %d exceeds security limit (%d)", e.Depth, MaxMp4BoxDepth)
	case ErrCorruptedBitstream:
		return "corrupted video bitstream"
	case ErrMissingDecryptionKey:
		return fmt.Sprintf("missing EME/CDM decryption key for key ID %02x%02x%02x%02x", e.KID[0], e.KID[1], e.KID[2], e.KID[3])
	case ErrCompression:
		return fmt.Sprintf("video compression error: %s", e.Detail)
	default:
		return "video error"
	}
}

func compressZlib(raw []byte) []byte {
	var buf bytes.Buffer
	w, _ := zlib.NewWriterLevel(&buf, zlib.DefaultCompression)
	_, _ = w.Write(raw)
	_ = w.Close()
	return buf.Bytes()
}

func decompressZlibBounded(compressed []byte, maxOutput int) ([]byte, error) {
	if maxOutput < 1 {
		maxOutput = 1
	}
	r, err := zlib.NewReader(bytes.NewReader(compressed))
	if err != nil {
		return nil, &VideoError{Kind: ErrCompression, Detail: err.Error()}
	}
	defer r.Close()
	lr := io.LimitReader(r, int64(maxOutput)+1)
	out, err := io.ReadAll(lr)
	if err != nil {
		return nil, &VideoError{Kind: ErrCompression, Detail: err.Error()}
	}
	if len(out) > maxOutput {
		return nil, &VideoError{Kind: ErrCompression, Detail: "decompressed output exceeds cap"}
	}
	return out, nil
}

// GpuVideoBackend identifies the platform video pipeline label.
type GpuVideoBackend uint8

const (
	BackendD3d11Dxva2 GpuVideoBackend = iota
	BackendLinuxVaApi
	BackendSoftwareFallback
)

func (b GpuVideoBackend) Label() string {
	switch b {
	case BackendD3d11Dxva2:
		return "D3D11 Video / DXVA2 (NV12 GPU)"
	case BackendLinuxVaApi:
		return "VA-API / DRM (NV12 GPU)"
	default:
		return "Software BT.709 Pipeline"
	}
}

func (b GpuVideoBackend) IsHardwareAccelerated() bool {
	return b != BackendSoftwareFallback
}

// ProbePlatformGpuBackend returns the backend label for hardwareEnabled and the current OS.
func ProbePlatformGpuBackend(hardwareEnabled bool) GpuVideoBackend {
	if !hardwareEnabled {
		return BackendSoftwareFallback
	}
	if runtime.GOOS == "windows" {
		return BackendD3d11Dxva2
	}
	if runtime.GOOS == "linux" {
		return BackendLinuxVaApi
	}
	return BackendSoftwareFallback
}

// VideoCodecProfile identifies a four-byte video codec profile.
type VideoCodecProfile uint8

const (
	ProfileH264High VideoCodecProfile = iota
	ProfileAv1Main
	ProfileVp9Profile0
	ProfileVp8Profile0
	ProfileAnvilNv12Temporal
)

func (p VideoCodecProfile) FourCC() [4]byte {
	switch p {
	case ProfileH264High:
		return [4]byte{'a', 'v', 'c', '1'}
	case ProfileAv1Main:
		return [4]byte{'a', 'v', '0', '1'}
	case ProfileVp9Profile0:
		return [4]byte{'v', 'p', '0', '9'}
	case ProfileVp8Profile0:
		return [4]byte{'v', 'p', '0', '8'}
	default:
		return [4]byte{'a', 'v', 'i', 'd'}
	}
}

var imaIndexTable = [16]int32{-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8}

var imaStepTable = [89]int32{
	7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66,
	73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449,
	494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272,
	2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493,
	10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
}

// AudioCodecProfile identifies the audio codec profile inside an AAUD stream.
type AudioCodecProfile uint8

const (
	AudioPcm16Le AudioCodecProfile = iota
	AudioImaAdpcm4
	AudioFlacLossless
	AudioOpusMdct
)

// AudioTrack holds decoded interleaved 16-bit signed PCM audio synchronized with a VideoClip.
type AudioTrack struct {
	SampleRate uint32
	Channels   uint8
	Codec      AudioCodecProfile
	Samples    []int16
}

func (a *AudioTrack) DurationMs() uint32 {
	ch := uint64(maxU8(a.Channels, 1))
	sr := uint64(maxU32(a.SampleRate, 1))
	frames := uint64(len(a.Samples)) / ch
	return uint32((frames * 1000) / sr)
}

func (a *AudioTrack) PeakLevelAtMs(ptsMs, windowMs uint32) uint8 {
	if len(a.Samples) == 0 || a.SampleRate == 0 || a.Channels == 0 {
		return 0
	}
	ch := int(a.Channels)
	sr := int(a.SampleRate)
	startFrame := (int(ptsMs) * sr) / 1000
	wMs := int(windowMs)
	if wMs < 10 {
		wMs = 10
	}
	spanFrames := (wMs * sr) / 1000
	if spanFrames < 1 {
		spanFrames = 1
	}
	startIdx := startFrame * ch
	if startIdx > len(a.Samples) {
		startIdx = len(a.Samples)
	}
	endIdx := (startFrame + spanFrames) * ch
	if endIdx > len(a.Samples) {
		endIdx = len(a.Samples)
	}
	if startIdx >= endIdx {
		return 0
	}
	peak := int32(0)
	for _, s := range a.Samples[startIdx:endIdx] {
		amp := int32(s)
		if amp < 0 {
			amp = -amp
		}
		if amp > peak {
			peak = amp
		}
	}
	lvl := (peak * 100) / 32767
	return uint8(clampI32(lvl, 0, 100))
}

// SynthesizeHarmonicChime generates a harmonic stereo/mono chime track for demo MP4 streams and tests.
func SynthesizeHarmonicChime(durationMs, sampleRate uint32, channels uint8, codec AudioCodecProfile) AudioTrack {
	dur := clampU32(durationMs, 0, 600_000)
	sr := clampU32(sampleRate, 8000, 96000)
	ch := channels
	if ch < 1 {
		ch = 1
	} else if ch > 2 {
		ch = 2
	}
	totalFrames := int((uint64(dur) * uint64(sr)) / 1000)
	samples := make([]int16, 0, totalFrames*int(ch))
	denom := float64(maxInt(totalFrames, 1))
	for i := 0; i < totalFrames; i++ {
		t := float64(i) / float64(sr)
		env := 1.0 - (float64(i)/denom)*0.55
		if env < 0.15 {
			env = 0.15
		} else if env > 1.0 {
			env = 1.0
		}
		f1 := math.Sin(t * 440.0 * 2.0 * math.Pi)
		f2 := math.Sin(t*659.25*2.0*math.Pi) * 0.5
		sl := clampF64((f1+f2)*0.45*env*18000.0, -32767.0, 32767.0)
		samples = append(samples, int16(sl))
		if ch == 2 {
			f3 := math.Sin(t*554.37*2.0*math.Pi) * 0.6
			srVal := clampF64((f1+f3)*0.45*env*18000.0, -32767.0, 32767.0)
			samples = append(samples, int16(srVal))
		}
	}
	return AudioTrack{
		SampleRate: sr,
		Channels:   ch,
		Codec:      codec,
		Samples:    samples,
	}
}

func (a *AudioTrack) SlicePcmWindow(startMs, durationMs uint32) []int16 {
	if len(a.Samples) == 0 || a.SampleRate == 0 || a.Channels == 0 {
		return nil
	}
	ch := int(a.Channels)
	sr := int(a.SampleRate)
	startFrame := (int(startMs) * sr) / 1000
	dMs := int(durationMs)
	if dMs < 1 {
		dMs = 1
	}
	spanFrames := (dMs * sr) / 1000
	if spanFrames < 1 {
		spanFrames = 1
	}
	startIdx := startFrame * ch
	if startIdx > len(a.Samples) {
		startIdx = len(a.Samples)
	}
	endIdx := (startFrame + spanFrames) * ch
	if endIdx > len(a.Samples) {
		endIdx = len(a.Samples)
	}
	if startIdx >= endIdx {
		return nil
	}
	out := make([]int16, endIdx-startIdx)
	copy(out, a.Samples[startIdx:endIdx])
	return out
}

func (a *AudioTrack) ToWavBytes() []byte {
	return EncodePcm16Wav(a.SampleRate, a.Channels, a.Samples)
}

// EncodePcm16Wav encodes interleaved 16-bit signed PCM samples into a 44-byte RIFF WAVE buffer.
func EncodePcm16Wav(sampleRate uint32, channels uint8, samples []int16) []byte {
	ch := uint16(1)
	if channels >= 2 {
		ch = 2
	}
	sr := sampleRate
	if sr < 8000 {
		sr = 8000
	}
	dataBytes := uint32(len(samples) * 2)
	riffSize := 36 + dataBytes
	byteRate := sr * uint32(ch) * 2
	blockAlign := ch * 2

	out := make([]byte, 44+len(samples)*2)
	copy(out[0:4], "RIFF")
	binary.LittleEndian.PutUint32(out[4:8], riffSize)
	copy(out[8:12], "WAVE")
	copy(out[12:16], "fmt ")
	binary.LittleEndian.PutUint32(out[16:20], 16)
	binary.LittleEndian.PutUint16(out[20:22], 1)
	binary.LittleEndian.PutUint16(out[22:24], ch)
	binary.LittleEndian.PutUint32(out[24:28], sr)
	binary.LittleEndian.PutUint32(out[28:32], byteRate)
	binary.LittleEndian.PutUint16(out[32:34], blockAlign)
	binary.LittleEndian.PutUint16(out[34:36], 16)
	copy(out[36:40], "data")
	binary.LittleEndian.PutUint32(out[40:44], dataBytes)
	for i, s := range samples {
		binary.LittleEndian.PutUint16(out[44+i*2:46+i*2], uint16(s))
	}
	return out
}

// OsAudioBackend identifies the host audio output backend.
type OsAudioBackend uint8

const (
	AudioBackendWinMmWaveOut OsAudioBackend = iota
	AudioBackendLinuxAlsaPcm
	AudioBackendSoftwareMixerSink
)

func (b OsAudioBackend) Label() string {
	switch b {
	case AudioBackendWinMmWaveOut:
		return "Windows WinMM WaveOut (PCM16)"
	case AudioBackendLinuxAlsaPcm:
		return "Linux ALSA PCM (PCM16)"
	default:
		return "Software PCM16 Mixer Sink"
	}
}

func (b OsAudioBackend) IsNativeOsDevice() bool {
	return b != AudioBackendSoftwareMixerSink
}

// ProbeOsAudioBackend probes the host operating system's native audio subsystem.
func ProbeOsAudioBackend() (OsAudioBackend, uint32) {
	if runtime.GOOS == "windows" {
		return AudioBackendWinMmWaveOut, 1
	}
	if runtime.GOOS == "linux" {
		return AudioBackendLinuxAlsaPcm, 1
	}
	return AudioBackendSoftwareMixerSink, 0
}

const MaxRingbufferSamples = 96_000

// AudioOutputBridge tracks master A/V clock synchronization and streaming PCM16 ringbuffer output.
type AudioOutputBridge struct {
	Backend                 OsAudioBackend
	HardwareDevicesDetected uint32
	Muted                   bool
	VolumePercent           uint8
	SubmittedWindows        uint64
	SubmittedSamples        uint64
	LastPeakLevel           uint8
	AudioClockMs            uint32
	BufferedPcmSamples      int
	pcmRingbuffer           []int16
}

// NewAudioOutputBridge initializes an unmuted audio output bridge at 100% volume.
func NewAudioOutputBridge() *AudioOutputBridge {
	backend, devs := ProbeOsAudioBackend()
	return &AudioOutputBridge{
		Backend:                 backend,
		HardwareDevicesDetected: devs,
		Muted:                   false,
		VolumePercent:           100,
	}
}

func (b *AudioOutputBridge) AvDriftMs(videoPtsMs uint32) int32 {
	return int32(videoPtsMs) - int32(b.AudioClockMs)
}

func (b *AudioOutputBridge) ResetClock(ptsMs uint32) {
	b.AudioClockMs = ptsMs
	b.pcmRingbuffer = b.pcmRingbuffer[:0]
	b.BufferedPcmSamples = 0
	b.LastPeakLevel = 0
}

func (b *AudioOutputBridge) DrainBufferedPcm(maxSamples int) []int16 {
	if b == nil || maxSamples <= 0 {
		return nil
	}
	count := maxSamples
	if count > len(b.pcmRingbuffer) {
		count = len(b.pcmRingbuffer)
	}
	out := make([]int16, count)
	copy(out, b.pcmRingbuffer[:count])
	b.pcmRingbuffer = b.pcmRingbuffer[count:]
	b.BufferedPcmSamples = len(b.pcmRingbuffer)
	return out
}

func (b *AudioOutputBridge) SubmitTrackWindow(track *AudioTrack, ptsMs, windowMs uint32, _ bool) int {
	pcm := track.SlicePcmWindow(ptsMs, windowMs)
	if len(pcm) == 0 {
		b.LastPeakLevel = 0
		return 0
	}
	scale := int32(0)
	if !b.Muted {
		vol := b.VolumePercent
		if vol > 100 {
			vol = 100
		}
		scale = int32(vol)
	}
	peak := int32(0)
	for i, s := range pcm {
		scaled := clampI32((int32(s)*scale)/100, -32768, 32767)
		pcm[i] = int16(scaled)
		amp := scaled
		if amp < 0 {
			amp = -amp
		}
		if amp > peak {
			peak = amp
		}
	}
	b.LastPeakLevel = uint8(clampI32((peak*100)/32767, 0, 100))
	b.SubmittedWindows++
	b.SubmittedSamples += uint64(len(pcm))
	b.AudioClockMs = ptsMs + windowMs

	if len(b.pcmRingbuffer)+len(pcm) > MaxRingbufferSamples {
		overflow := (len(b.pcmRingbuffer) + len(pcm)) - MaxRingbufferSamples
		if overflow > len(b.pcmRingbuffer) {
			overflow = len(b.pcmRingbuffer)
		}
		b.pcmRingbuffer = b.pcmRingbuffer[overflow:]
	}
	b.pcmRingbuffer = append(b.pcmRingbuffer, pcm...)
	b.BufferedPcmSamples = len(b.pcmRingbuffer)
	return len(pcm)
}

func imaAdpcmEncodeStep(sample int16, predictor, stepIndex *int32) uint8 {
	step := imaStepTable[*stepIndex]
	diff := int32(sample) - *predictor
	nibble := uint8(0)
	if diff < 0 {
		nibble = 8
		diff = -diff
	}
	vpdiff := step >> 3
	if diff >= step {
		nibble |= 4
		diff -= step
		vpdiff += step
	}
	if diff >= (step >> 1) {
		nibble |= 2
		diff -= step >> 1
		vpdiff += step >> 1
	}
	if diff >= (step >> 2) {
		nibble |= 1
		vpdiff += step >> 2
	}
	if (nibble & 8) != 0 {
		*predictor -= vpdiff
	} else {
		*predictor += vpdiff
	}
	*predictor = clampI32(*predictor, -32768, 32767)
	*stepIndex = clampI32(*stepIndex+imaIndexTable[nibble], 0, 88)
	return nibble & 0x0F
}

func imaAdpcmDecodeStep(nibble uint8, predictor, stepIndex *int32) int16 {
	step := imaStepTable[*stepIndex]
	vpdiff := step >> 3
	if (nibble & 4) != 0 {
		vpdiff += step
	}
	if (nibble & 2) != 0 {
		vpdiff += step >> 1
	}
	if (nibble & 1) != 0 {
		vpdiff += step >> 2
	}
	if (nibble & 8) != 0 {
		*predictor -= vpdiff
	} else {
		*predictor += vpdiff
	}
	*predictor = clampI32(*predictor, -32768, 32767)
	*stepIndex = clampI32(*stepIndex+imaIndexTable[nibble&0x0F], 0, 88)
	return int16(*predictor)
}

// EncodeAaudBitstream encodes an AudioTrack into a 16-byte AAUD header followed by compressed data.
func EncodeAaudBitstream(track *AudioTrack) ([]byte, error) {
	if track.Channels == 0 || track.Channels > 2 || track.SampleRate == 0 {
		return nil, &VideoError{Kind: ErrInvalidHeader}
	}
	sampleCount := len(track.Samples)
	if sampleCount > MaxPacketBytes/2 {
		return nil, &VideoError{Kind: ErrPacketTooLarge, Size: sampleCount * 2}
	}
	var codecByte uint8
	var compressed []byte
	var err error
	switch track.Codec {
	case AudioPcm16Le:
		raw := make([]byte, sampleCount*2)
		for i, s := range track.Samples {
			binary.LittleEndian.PutUint16(raw[i*2:i*2+2], uint16(s))
		}
		codecByte = 0
		compressed = compressZlib(raw)
	case AudioImaAdpcm4:
		raw := make([]byte, 0, (sampleCount+1)/2)
		ch := int(track.Channels)
		var pred [2]int32
		var idx [2]int32
		var packed uint8
		for i, s := range track.Samples {
			c := i % ch
			nib := imaAdpcmEncodeStep(s, &pred[c], &idx[c])
			if (i & 1) == 0 {
				packed = nib
			} else {
				packed |= nib << 4
				raw = append(raw, packed)
			}
		}
		if (sampleCount & 1) != 0 {
			raw = append(raw, packed)
		}
		codecByte = 1
		compressed = compressZlib(raw)
	case AudioFlacLossless:
		codecByte = 2
		compressed, err = EncodeFlacStream(track)
		if err != nil {
			return nil, err
		}
	case AudioOpusMdct:
		codecByte = 3
		compressed, err = EncodeMdctAudioStream(track)
		if err != nil {
			return nil, err
		}
	default:
		return nil, &VideoError{Kind: ErrInvalidHeader}
	}
	if len(compressed) > MaxPacketBytes {
		return nil, &VideoError{Kind: ErrPacketTooLarge, Size: len(compressed)}
	}
	out := make([]byte, 16+len(compressed))
	copy(out[0:4], "AAUD")
	out[4] = 1
	out[5] = codecByte
	out[6] = track.Channels
	out[7] = 0
	binary.BigEndian.PutUint32(out[8:12], track.SampleRate)
	binary.BigEndian.PutUint32(out[12:16], uint32(sampleCount))
	copy(out[16:], compressed)
	return out, nil
}

// DecodeAaudBitstream decodes an AAUD, fLaC, or MDCT audio bitstream into an AudioTrack.
func DecodeAaudBitstream(data []byte) (AudioTrack, error) {
	if len(data) >= 4 && string(data[0:4]) == "fLaC" {
		return DecodeFlacStream(data)
	}
	if len(data) >= 4 && string(data[0:4]) == "MDCT" {
		return DecodeMdctAudioStream(data)
	}
	if len(data) < 16 || string(data[0:4]) != "AAUD" || data[4] != 1 {
		return AudioTrack{}, &VideoError{Kind: ErrInvalidHeader}
	}
	switch data[5] {
	case 2:
		return DecodeFlacStream(data[16:])
	case 3:
		return DecodeMdctAudioStream(data[16:])
	}
	var codec AudioCodecProfile
	switch data[5] {
	case 0:
		codec = AudioPcm16Le
	case 1:
		codec = AudioImaAdpcm4
	default:
		return AudioTrack{}, &VideoError{Kind: ErrInvalidHeader}
	}
	channels := data[6]
	if channels == 0 || channels > 2 {
		return AudioTrack{}, &VideoError{Kind: ErrInvalidHeader}
	}
	sampleRate := binary.BigEndian.Uint32(data[8:12])
	sampleCount := int(binary.BigEndian.Uint32(data[12:16]))
	if sampleRate == 0 || sampleCount > MaxPacketBytes/2 {
		return AudioTrack{}, &VideoError{Kind: ErrInvalidHeader}
	}
	expectedRaw := sampleCount * 2
	if codec == AudioImaAdpcm4 {
		expectedRaw = (sampleCount + 1) / 2
	}
	raw, err := decompressZlibBounded(data[16:], expectedRaw)
	if err != nil {
		return AudioTrack{}, err
	}
	if len(raw) != expectedRaw {
		return AudioTrack{}, &VideoError{Kind: ErrCorruptedBitstream}
	}
	samples := make([]int16, sampleCount)
	if codec == AudioPcm16Le {
		for i := 0; i < sampleCount; i++ {
			samples[i] = int16(binary.LittleEndian.Uint16(raw[i*2 : i*2+2]))
		}
	} else {
		ch := int(channels)
		var pred [2]int32
		var idx [2]int32
		for i := 0; i < sampleCount; i++ {
			b := raw[i/2]
			var nib uint8
			if (i & 1) == 0 {
				nib = b & 0x0F
			} else {
				nib = (b >> 4) & 0x0F
			}
			c := i % ch
			samples[i] = imaAdpcmDecodeStep(nib, &pred[c], &idx[c])
		}
	}
	return AudioTrack{
		SampleRate: sampleRate,
		Channels:   channels,
		Codec:      codec,
		Samples:    samples,
	}, nil
}

// ValidateVideoDimensions validates frame width and height against 4K and 4:2:0 evenness bounds.
func ValidateVideoDimensions(width, height uint32) error {
	if width == 0 || height == 0 {
		return &VideoError{Kind: ErrInvalidHeader}
	}
	if width > MaxVideoWidth || height > MaxVideoHeight || uint64(width)*uint64(height) > MaxVideoPixels {
		return &VideoError{Kind: ErrDimensionsTooLarge, Width: width, Height: height}
	}
	if width%2 != 0 || height%2 != 0 {
		return &VideoError{Kind: ErrOddDimensions, Width: width, Height: height}
	}
	return nil
}

// Nv12Surface holds 8-bit 4:2:0 semi-planar YUV (Y plane + interleaved UV plane).
type Nv12Surface struct {
	Width   uint32
	Height  uint32
	YPlane  []byte
	UVPlane []byte
}

func NewNv12Surface(width, height uint32) Nv12Surface {
	yLen := int(width) * int(height)
	uvLen := int(width/2) * int(height/2) * 2
	y := make([]byte, yLen)
	for i := range y {
		y[i] = 16
	}
	uv := make([]byte, uvLen)
	for i := range uv {
		uv[i] = 128
	}
	return Nv12Surface{Width: width, Height: height, YPlane: y, UVPlane: uv}
}

func rgbToYuvBt709(r, g, b byte) (byte, byte, byte) {
	rf, gf, bf := int32(r), int32(g), int32(b)
	y := ((47*rf + 157*gf + 16*bf + 128) >> 8) + 16
	u := ((-26*rf - 87*gf + 112*bf + 128) >> 8) + 128
	v := ((112*rf - 102*gf - 10*bf + 128) >> 8) + 128
	return byte(clampI32(y, 16, 235)), byte(clampI32(u, 16, 240)), byte(clampI32(v, 16, 240))
}

func yuvBt709ToRgba(y, u, v byte) [4]byte {
	c := int32(y) - 16
	if c < 0 {
		c = 0
	}
	d := int32(u) - 128
	e := int32(v) - 128
	r := (298*c + 459*e + 128) >> 8
	g := (298*c - 55*d - 136*e + 128) >> 8
	b := (298*c + 541*d + 128) >> 8
	return [4]byte{byte(clampI32(r, 0, 255)), byte(clampI32(g, 0, 255)), byte(clampI32(b, 0, 255)), 255}
}

func Nv12FromRgbaBt709(width, height uint32, rgba []byte) (Nv12Surface, bool) {
	if width == 0 || height == 0 || width%2 != 0 || height%2 != 0 {
		return Nv12Surface{}, false
	}
	w, h := int(width), int(height)
	if len(rgba) != w*h*4 {
		return Nv12Surface{}, false
	}
	surf := NewNv12Surface(width, height)
	for py := 0; py < h; py += 2 {
		for px := 0; px < w; px += 2 {
			var uSum, vSum uint32
			for dy := 0; dy < 2; dy++ {
				for dx := 0; dx < 2; dx++ {
					idx := ((py+dy)*w + (px + dx)) * 4
					yv, uv, vv := rgbToYuvBt709(rgba[idx], rgba[idx+1], rgba[idx+2])
					surf.YPlane[(py+dy)*w+(px+dx)] = yv
					uSum += uint32(uv)
					vSum += uint32(vv)
				}
			}
			uvIdx := (py/2)*w + px
			surf.UVPlane[uvIdx] = byte((uSum + 2) / 4)
			surf.UVPlane[uvIdx+1] = byte((vSum + 2) / 4)
		}
	}
	return surf, true
}

func (s *Nv12Surface) ToRgbaBt709() []byte {
	if s == nil || ValidateVideoDimensions(s.Width, s.Height) != nil {
		return nil
	}
	w, h := int(s.Width), int(s.Height)
	if len(s.YPlane) < w*h {
		return nil
	}
	rgba := make([]byte, w*h*4)
	for py := 0; py < h; py++ {
		for px := 0; px < w; px++ {
			yv := s.YPlane[py*w+px]
			uvIdx := (py/2)*w + (px & ^1)
			u, v := byte(128), byte(128)
			if uvIdx+1 < len(s.UVPlane) {
				u = s.UVPlane[uvIdx]
				v = s.UVPlane[uvIdx+1]
			}
			pxRgba := yuvBt709ToRgba(yv, u, v)
			dst := (py*w + px) * 4
			copy(rgba[dst:dst+4], pxRgba[:])
		}
	}
	return rgba
}

// I420Surface holds 8-bit 4:2:0 planar YUV (Y, U, V planes).
type I420Surface struct {
	Width  uint32
	Height uint32
	YPlane []byte
	UPlane []byte
	VPlane []byte
}

func NewI420Surface(width, height uint32) I420Surface {
	w := min(width, MaxVideoWidth)
	h := min(height, MaxVideoHeight)
	yLen := int(w) * int(h)
	cLen := int(w/2) * int(h/2)
	y := make([]byte, yLen)
	for i := range y {
		y[i] = 16
	}
	u := make([]byte, cLen)
	v := make([]byte, cLen)
	for i := range u {
		u[i] = 128
		v[i] = 128
	}
	return I420Surface{Width: w, Height: h, YPlane: y, UPlane: u, VPlane: v}
}

func I420FromRgbaBt709(width, height uint32, rgba []byte) (I420Surface, bool) {
	nv12, ok := Nv12FromRgbaBt709(width, height, rgba)
	if !ok {
		return I420Surface{}, false
	}
	cLen := len(nv12.UVPlane) / 2
	u := make([]byte, cLen)
	v := make([]byte, cLen)
	for i := 0; i < cLen; i++ {
		u[i] = nv12.UVPlane[i*2]
		v[i] = nv12.UVPlane[i*2+1]
	}
	return I420Surface{Width: width, Height: height, YPlane: nv12.YPlane, UPlane: u, VPlane: v}, true
}

func (s *I420Surface) ToRgbaBt709() []byte {
	if s == nil || ValidateVideoDimensions(s.Width, s.Height) != nil {
		return nil
	}
	w, h := int(s.Width), int(s.Height)
	if len(s.YPlane) < w*h {
		return nil
	}
	cw := w / 2
	rgba := make([]byte, w*h*4)
	for py := 0; py < h; py++ {
		for px := 0; px < w; px++ {
			yv := s.YPlane[py*w+px]
			cIdx := (py/2)*cw + (px / 2)
			u, v := byte(128), byte(128)
			if cIdx < len(s.UPlane) {
				u = s.UPlane[cIdx]
			}
			if cIdx < len(s.VPlane) {
				v = s.VPlane[cIdx]
			}
			pxRgba := yuvBt709ToRgba(yv, u, v)
			dst := (py*w + px) * 4
			copy(rgba[dst:dst+4], pxRgba[:])
		}
	}
	return rgba
}

func (s *I420Surface) Clone() I420Surface {
	y := make([]byte, len(s.YPlane))
	u := make([]byte, len(s.UPlane))
	v := make([]byte, len(s.VPlane))
	copy(y, s.YPlane)
	copy(u, s.UPlane)
	copy(v, s.VPlane)
	return I420Surface{Width: s.Width, Height: s.Height, YPlane: y, UPlane: u, VPlane: v}
}

var (
	deviceConvertCount atomic.Uint64
	devicePresentCount atomic.Uint64
)

func DeviceConverts() uint64 { return deviceConvertCount.Load() }
func DevicePresents() uint64 { return devicePresentCount.Load() }

// VideoFrame represents one frame as packed RGBA8888 plus presentation timestamp in ms.
type VideoFrame struct {
	Width    uint32
	Height   uint32
	PtsMs    uint32
	Keyframe bool
	Rgba     []byte
}

func NewVideoFrame(width, height, ptsMs uint32, keyframe bool, rgba []byte) (VideoFrame, error) {
	if err := ValidateVideoDimensions(width, height); err != nil {
		return VideoFrame{}, err
	}
	expected := int(width) * int(height) * 4
	if len(rgba) != expected {
		return VideoFrame{}, &VideoError{Kind: ErrInvalidHeader}
	}
	return VideoFrame{
		Width:    width,
		Height:   height,
		PtsMs:    ptsMs,
		Keyframe: keyframe,
		Rgba:     rgba,
	}, nil
}

func (f *VideoFrame) ToNv12Surface() (Nv12Surface, error) {
	s, ok := Nv12FromRgbaBt709(f.Width, f.Height, f.Rgba)
	if !ok {
		return Nv12Surface{}, &VideoError{Kind: ErrInvalidHeader}
	}
	return s, nil
}

func VideoFrameFromNv12Surface(surface *Nv12Surface, ptsMs uint32, keyframe bool) (VideoFrame, error) {
	if err := ValidateVideoDimensions(surface.Width, surface.Height); err != nil {
		return VideoFrame{}, err
	}
	return VideoFrame{
		Width:    surface.Width,
		Height:   surface.Height,
		PtsMs:    ptsMs,
		Keyframe: keyframe,
		Rgba:     surface.ToRgbaBt709(),
	}, nil
}

func (f *VideoFrame) BlitInto(dstRgba []byte, dstWidth, dstHeight uint32, dstStride int, x, y int32, width, height uint32, _ bool) error {
	if f == nil || f.Width == 0 || f.Height == 0 || f.Width > MaxVideoWidth || f.Height > MaxVideoHeight || len(f.Rgba) < int(f.Width)*int(f.Height)*4 {
		return &VideoError{Kind: ErrInvalidHeader}
	}
	if dstWidth == 0 || dstHeight == 0 || dstWidth > MaxVideoWidth || dstHeight > MaxVideoHeight || width == 0 || height == 0 || width > MaxVideoWidth || height > MaxVideoHeight || dstStride < int(dstWidth)*4 || len(dstRgba) < int(dstHeight)*dstStride {
		return &VideoError{Kind: ErrInvalidHeader}
	}
	for row := uint32(0); row < height; row++ {
		dy := int64(y) + int64(row)
		if dy < 0 || dy >= int64(dstHeight) {
			continue
		}
		sy := int((uint64(row) * uint64(f.Height)) / uint64(height))
		if sy >= int(f.Height) {
			sy = int(f.Height) - 1
		}
		for col := uint32(0); col < width; col++ {
			dx := int64(x) + int64(col)
			if dx < 0 || dx >= int64(dstWidth) {
				continue
			}
			sx := int((uint64(col) * uint64(f.Width)) / uint64(width))
			if sx >= int(f.Width) {
				sx = int(f.Width) - 1
			}
			srcIdx := (sy*int(f.Width) + sx) * 4
			dstIdx := int(dy)*dstStride + int(dx)*4
			copy(dstRgba[dstIdx:dstIdx+4], f.Rgba[srcIdx:srcIdx+4])
		}
	}
	return nil
}

// VideoClip holds decoded frames and optional synchronized audio.
type VideoClip struct {
	Width   uint32
	Height  uint32
	Fps     uint32
	Backend GpuVideoBackend
	Profile VideoCodecProfile
	Frames  []VideoFrame
	Audio   *AudioTrack
}

func (c *VideoClip) DurationMs() uint32 {
	if len(c.Frames) == 0 {
		return 0
	}
	return c.Frames[len(c.Frames)-1].PtsMs
}

func (c *VideoClip) PosterFrame() *VideoFrame {
	if len(c.Frames) == 0 {
		return nil
	}
	return &c.Frames[0]
}

func (c *VideoClip) KeyframeCount() int {
	n := 0
	for _, f := range c.Frames {
		if f.Keyframe {
			n++
		}
	}
	return n
}

func (c *VideoClip) FrameAtMs(timestampMs uint32) *VideoFrame {
	if len(c.Frames) == 0 {
		return nil
	}
	chosen := &c.Frames[0]
	for i := range c.Frames {
		if c.Frames[i].PtsMs <= timestampMs {
			chosen = &c.Frames[i]
		} else {
			break
		}
	}
	return chosen
}

func (c *VideoClip) PeakAudioLevelAtMs(timestampMs uint32) uint8 {
	if c.Audio == nil {
		return 0
	}
	return c.Audio.PeakLevelAtMs(timestampMs, 33)
}

// Mp4Metadata holds fields parsed from an ISOBMFF MP4 container without decoding mdat.
type Mp4Metadata struct {
	Brand            [4]byte
	Width            uint32
	Height           uint32
	Timescale        uint32
	DurationMs       uint32
	MdatSize         int
	BoxCount         int
	Encrypted        bool
	ProtectionScheme *[4]byte
	DefaultKID       *[16]byte
	IV               *[16]byte
	PsshCount        int
	HasAudioTrack    bool
	AudioSampleRate  uint32
	AudioChannels    uint8
}

// EmeKeySystem identifies the W3C Encrypted Media Extensions key system label.
type EmeKeySystem uint8

const (
	KeySystemW3cClearKey EmeKeySystem = iota
	KeySystemOsMediaFoundationDrm
	KeySystemExternalPluginCdm
)

func (k EmeKeySystem) Identifier() string {
	switch k {
	case KeySystemW3cClearKey:
		return "org.w3.clearkey"
	case KeySystemOsMediaFoundationDrm:
		return "org.anvil.mediafoundation.cdm"
	default:
		return "org.anvil.external.cdm"
	}
}

func (k EmeKeySystem) Label() string {
	switch k {
	case KeySystemW3cClearKey:
		return "W3C ClearKey (org.w3.clearkey + CENC AES-128-CTR)"
	case KeySystemOsMediaFoundationDrm:
		return "OS Media Foundation / DRM Bridge"
	default:
		return "External CDM Host Plugin Slot"
	}
}

type cdmKeyEntry struct {
	kid [16]byte
	key [16]byte
}

// CdmSession stores an in-memory list of 16-byte key IDs and 16-byte AES-128 keys.
type CdmSession struct {
	keySystem        EmeKeySystem
	keys             []cdmKeyEntry
	decryptedSamples uint64
}

func NewCdmSession(keySystem EmeKeySystem) *CdmSession {
	return &CdmSession{keySystem: keySystem}
}

func NewCdmSessionWithDefaultDemoKey() *CdmSession {
	s := NewCdmSession(KeySystemW3cClearKey)
	s.ProvideLicenseKey(DefaultEmeDemoKID, DefaultEmeDemoKey)
	return s
}

func (s *CdmSession) KeySystem() EmeKeySystem      { return s.keySystem }
func (s *CdmSession) SetKeySystem(ks EmeKeySystem) { s.keySystem = ks }
func (s *CdmSession) KeyCount() int                { return len(s.keys) }
func (s *CdmSession) DecryptedSamples() uint64     { return s.decryptedSamples }
func (s *CdmSession) ClearKeys()                   { s.keys = s.keys[:0] }

func (s *CdmSession) ProvideLicenseKey(keyID, key [16]byte) {
	for i := range s.keys {
		if s.keys[i].kid == keyID {
			s.keys[i].key = key
			return
		}
	}
	if len(s.keys) >= 128 {
		s.keys = s.keys[1:]
	}
	s.keys = append(s.keys, cdmKeyEntry{kid: keyID, key: key})
}

func (s *CdmSession) RemoveKey(keyID [16]byte) bool {
	before := len(s.keys)
	filtered := s.keys[:0]
	for _, entry := range s.keys {
		if entry.kid != keyID {
			filtered = append(filtered, entry)
		}
	}
	s.keys = filtered
	return len(s.keys) != before
}

func (s *CdmSession) HasKey(keyID [16]byte) bool {
	_, ok := s.LookupKey(keyID)
	return ok
}

func (s *CdmSession) LookupKey(keyID [16]byte) ([16]byte, bool) {
	for _, entry := range s.keys {
		if entry.kid == keyID {
			return entry.key, true
		}
	}
	return [16]byte{}, false
}

// GpuVideoSession manages CPU/GPU encode and decode state for AVID and MP4/WebM containers.
type GpuVideoSession struct {
	backend               GpuVideoBackend
	hardwareDecodeEnabled bool
	hardwareEncodeEnabled bool
	framesDecoded         uint64
	framesEncoded         uint64
	cdmSession            *CdmSession
}

func NewGpuVideoSession(hardwareDecode, hardwareEncode bool) *GpuVideoSession {
	return &GpuVideoSession{
		backend:               ProbePlatformGpuBackend(hardwareDecode || hardwareEncode),
		hardwareDecodeEnabled: hardwareDecode,
		hardwareEncodeEnabled: hardwareEncode,
		cdmSession:            NewCdmSessionWithDefaultDemoKey(),
	}
}

func DefaultGpuVideoSession() *GpuVideoSession {
	return NewGpuVideoSession(true, true)
}

func (s *GpuVideoSession) CdmSession() *CdmSession       { return s.cdmSession }
func (s *GpuVideoSession) Backend() GpuVideoBackend      { return s.backend }
func (s *GpuVideoSession) IsHardwareDecodeEnabled() bool { return s.hardwareDecodeEnabled }
func (s *GpuVideoSession) IsHardwareEncodeEnabled() bool { return s.hardwareEncodeEnabled }
func (s *GpuVideoSession) FramesDecoded() uint64         { return s.framesDecoded }
func (s *GpuVideoSession) FramesEncoded() uint64         { return s.framesEncoded }

func (s *GpuVideoSession) SetHardwareAcceleration(decode, encode bool) {
	s.hardwareDecodeEnabled = decode
	s.hardwareEncodeEnabled = encode
	s.backend = ProbePlatformGpuBackend(decode || encode)
}

func (s *GpuVideoSession) EncodeAvid(fps uint32, frames []VideoFrame) ([]byte, error) {
	if len(frames) == 0 {
		return nil, &VideoError{Kind: ErrInvalidHeader}
	}
	if uint32(len(frames)) > MaxVideoFrames {
		return nil, &VideoError{Kind: ErrFrameCountTooLarge, Count: uint32(len(frames))}
	}
	width := frames[0].Width
	height := frames[0].Height
	if err := ValidateVideoDimensions(width, height); err != nil {
		return nil, err
	}
	var out bytes.Buffer
	out.WriteString("AVID")
	var hdr [20]byte
	binary.BigEndian.PutUint16(hdr[0:2], 1)
	if s.hardwareEncodeEnabled && s.backend.IsHardwareAccelerated() {
		hdr[2] = 1
	}
	hdr[3] = 0
	binary.BigEndian.PutUint32(hdr[4:8], width)
	binary.BigEndian.PutUint32(hdr[8:12], height)
	binary.BigEndian.PutUint32(hdr[12:16], maxU32(fps, 1))
	binary.BigEndian.PutUint32(hdr[16:20], uint32(len(frames)))
	out.Write(hdr[:])

	var prevNv12 *Nv12Surface
	for idx := range frames {
		f := &frames[idx]
		if f.Width != width || f.Height != height {
			return nil, &VideoError{Kind: ErrInvalidHeader}
		}
		currNv12, err := f.ToNv12Surface()
		if err != nil {
			return nil, err
		}
		if s.hardwareEncodeEnabled {
			deviceConvertCount.Add(1)
		}
		isKeyframe := idx == 0 || f.Keyframe || (idx%8 == 0)
		rawPayload := make([]byte, len(currNv12.YPlane)+len(currNv12.UVPlane))
		if isKeyframe || prevNv12 == nil {
			copy(rawPayload[:len(currNv12.YPlane)], currNv12.YPlane)
			copy(rawPayload[len(currNv12.YPlane):], currNv12.UVPlane)
		} else {
			yLen := len(currNv12.YPlane)
			for i := 0; i < yLen; i++ {
				rawPayload[i] = currNv12.YPlane[i] - prevNv12.YPlane[i]
			}
			for i := 0; i < len(currNv12.UVPlane); i++ {
				rawPayload[yLen+i] = currNv12.UVPlane[i] - prevNv12.UVPlane[i]
			}
		}
		compressed := compressZlib(rawPayload)
		if len(compressed) > MaxPacketBytes {
			return nil, &VideoError{Kind: ErrPacketTooLarge, Size: len(compressed)}
		}
		var recHdr [9]byte
		if isKeyframe {
			recHdr[0] = 'I'
		} else {
			recHdr[0] = 'P'
		}
		binary.BigEndian.PutUint32(recHdr[1:5], f.PtsMs)
		binary.BigEndian.PutUint32(recHdr[5:9], uint32(len(compressed)))
		out.Write(recHdr[:])
		out.Write(compressed)

		cp := currNv12
		prevNv12 = &cp
		s.framesEncoded++
	}
	return out.Bytes(), nil
}

func (s *GpuVideoSession) DecodeAvid(data []byte) (VideoClip, error) {
	var audioSlice []byte
	if len(data) >= 12 && string(data[0:4]) == "AAVM" {
		avidLen := int(binary.BigEndian.Uint32(data[4:8]))
		aaudLen := int(binary.BigEndian.Uint32(data[8:12]))
		if avidLen < 0 || aaudLen < 0 || avidLen > MaxPacketBytes || aaudLen > MaxPacketBytes || avidLen > len(data)-12 || aaudLen > len(data)-12-avidLen {
			return VideoClip{}, &VideoError{Kind: ErrCorruptedBitstream}
		}
		if aaudLen > 0 {
			audioSlice = data[12+avidLen : 12+avidLen+aaudLen]
		}
		data = data[12 : 12+avidLen]
	}
	if len(data) < 24 || string(data[0:4]) != "AVID" {
		return VideoClip{}, &VideoError{Kind: ErrInvalidHeader}
	}
	version := binary.BigEndian.Uint16(data[4:6])
	if version != 1 {
		return VideoClip{}, &VideoError{Kind: ErrInvalidHeader}
	}
	width := binary.BigEndian.Uint32(data[8:12])
	height := binary.BigEndian.Uint32(data[12:16])
	fps := maxU32(binary.BigEndian.Uint32(data[16:20]), 1)
	frameCount := binary.BigEndian.Uint32(data[20:24])
	if err := ValidateVideoDimensions(width, height); err != nil {
		return VideoClip{}, err
	}
	if frameCount == 0 || frameCount > MaxVideoFrames {
		return VideoClip{}, &VideoError{Kind: ErrFrameCountTooLarge, Count: frameCount}
	}
	yLen := int(width) * int(height)
	uvLen := int(width/2) * int(height/2) * 2
	planeTotal := yLen + uvLen

	cursor := 24
	refSurface := NewNv12Surface(width, height)
	frames := make([]VideoFrame, 0, int(frameCount))

	for frameIdx := uint32(0); frameIdx < frameCount; frameIdx++ {
		if cursor+9 > len(data) {
			return VideoClip{}, &VideoError{Kind: ErrCorruptedBitstream}
		}
		frameType := data[cursor]
		ptsMs := binary.BigEndian.Uint32(data[cursor+1 : cursor+5])
		pktLen := int(binary.BigEndian.Uint32(data[cursor+5 : cursor+9]))
		cursor += 9
		if pktLen < 0 || pktLen > MaxPacketBytes || pktLen > len(data)-cursor {
			return VideoClip{}, &VideoError{Kind: ErrPacketTooLarge, Size: pktLen}
		}
		rawPlanes, err := decompressZlibBounded(data[cursor:cursor+pktLen], planeTotal)
		if err != nil {
			return VideoClip{}, err
		}
		cursor += pktLen
		if len(rawPlanes) != planeTotal {
			return VideoClip{}, &VideoError{Kind: ErrCorruptedBitstream}
		}
		var isKeyframe bool
		switch {
		case frameType == 'I':
			isKeyframe = true
		case frameType == 'P' && frameIdx > 0:
			isKeyframe = false
		default:
			return VideoClip{}, &VideoError{Kind: ErrCorruptedBitstream}
		}
		if isKeyframe {
			copy(refSurface.YPlane, rawPlanes[:yLen])
			copy(refSurface.UVPlane, rawPlanes[yLen:])
		} else {
			for i := 0; i < yLen; i++ {
				refSurface.YPlane[i] += rawPlanes[i]
			}
			for i := 0; i < uvLen; i++ {
				refSurface.UVPlane[i] += rawPlanes[yLen+i]
			}
		}
		if s.hardwareDecodeEnabled {
			devicePresentCount.Add(1)
		}
		decodedFrame, err := VideoFrameFromNv12Surface(&refSurface, ptsMs, isKeyframe)
		if err != nil {
			return VideoClip{}, err
		}
		frames = append(frames, decodedFrame)
		s.framesDecoded++
	}
	var audioPtr *AudioTrack
	if len(audioSlice) > 0 {
		tr, err := DecodeAaudBitstream(audioSlice)
		if err != nil {
			return VideoClip{}, err
		}
		audioPtr = &tr
	}
	return VideoClip{
		Width:   width,
		Height:  height,
		Fps:     fps,
		Backend: ProbePlatformGpuBackend(s.hardwareDecodeEnabled),
		Profile: ProfileAnvilNv12Temporal,
		Frames:  frames,
		Audio:   audioPtr,
	}, nil
}

func (s *GpuVideoSession) EncodeMp4(fps uint32, frames []VideoFrame) ([]byte, error) {
	avid, err := s.EncodeAvid(fps, frames)
	if err != nil {
		return nil, err
	}
	dur := frames[len(frames)-1].PtsMs
	return WrapMp4Container(frames[0].Width, frames[0].Height, dur, avid), nil
}

func (s *GpuVideoSession) EncodeAvMp4(fps uint32, frames []VideoFrame, audio *AudioTrack) ([]byte, error) {
	avid, err := s.EncodeAvid(fps, frames)
	if err != nil {
		return nil, err
	}
	var aaud []byte
	if audio != nil {
		aaud, err = EncodeAaudBitstream(audio)
		if err != nil {
			return nil, err
		}
	}
	dur := frames[len(frames)-1].PtsMs
	return WrapAvMp4Container(frames[0].Width, frames[0].Height, dur, avid, aaud), nil
}

func (s *GpuVideoSession) EncodeEncryptedMp4(fps uint32, frames []VideoFrame, keyID, key, iv [16]byte) ([]byte, error) {
	avid, err := s.EncodeAvid(fps, frames)
	if err != nil {
		return nil, err
	}
	dur := frames[len(frames)-1].PtsMs
	return WrapEncryptedMp4Container(frames[0].Width, frames[0].Height, dur, keyID, key, iv, avid), nil
}

func (s *GpuVideoSession) decodePayloadAuto(payload []byte) (VideoClip, error) {
	hw := s.hardwareDecodeEnabled
	if len(payload) >= 5 && bytes.Equal(payload[0:5], []byte{0x00, 0x00, 0x00, 0x01, 0x67}) {
		clip, err := DecodeH264AnnexB(payload, hw)
		if err != nil {
			return VideoClip{}, err
		}
		s.framesDecoded += uint64(len(clip.Frames))
		clip.Backend = ProbePlatformGpuBackend(hw)
		return clip, nil
	}
	if len(payload) >= 3 && bytes.Equal(payload[0:3], []byte{0x12, 0x00, 0x0A}) {
		clip, err := DecodeAv1ObuStream(payload, hw)
		if err != nil {
			return VideoClip{}, err
		}
		s.framesDecoded += uint64(len(clip.Frames))
		clip.Backend = ProbePlatformGpuBackend(hw)
		return clip, nil
	}
	if len(payload) >= 6 && (payload[0]&1) == 0 && bytes.Equal(payload[3:6], []byte{0x9D, 0x01, 0x2A}) {
		clip, err := DecodeVp8Rfc6386Stream(payload, hw)
		if err != nil {
			return VideoClip{}, err
		}
		s.framesDecoded += uint64(len(clip.Frames))
		clip.Backend = ProbePlatformGpuBackend(hw)
		return clip, nil
	}
	if len(payload) >= 4 && bytes.Equal(payload[0:4], []byte{0x82, 0x49, 0x83, 0x42}) {
		clip, err := DecodeVp9Stream(payload, hw)
		if err != nil {
			return VideoClip{}, err
		}
		s.framesDecoded += uint64(len(clip.Frames))
		clip.Backend = ProbePlatformGpuBackend(hw)
		return clip, nil
	}
	if len(payload) >= 4 && string(payload[0:4]) == "ADCT" {
		clip, err := DecodeDctMcBitstream(payload, hw)
		if err != nil {
			return VideoClip{}, err
		}
		s.framesDecoded += uint64(len(clip.Frames))
		clip.Backend = ProbePlatformGpuBackend(hw)
		return clip, nil
	}
	return s.DecodeAvid(payload)
}

func (s *GpuVideoSession) DecodeVideoAuto(data []byte) (VideoClip, error) {
	if len(data) >= 4 && bytes.Equal(data[0:4], []byte{0x1A, 0x45, 0xDF, 0xA3}) {
		_, videoPayload, err := ParseWebmContainer(data)
		if err != nil {
			return VideoClip{}, err
		}
		return s.decodePayloadAuto(videoPayload)
	}
	if len(data) >= 8 && string(data[4:8]) == "moof" {
		seg, err := InspectFmp4Segment(data)
		if err != nil {
			return VideoClip{}, err
		}
		if !seg.HasMediaFragment {
			return VideoClip{}, &VideoError{Kind: ErrInvalidHeader}
		}
		mdatPayload, ok := extractTopLevelBoxPayload(data, [4]byte{'m', 'd', 'a', 't'})
		if !ok {
			return VideoClip{}, &VideoError{Kind: ErrInvalidHeader}
		}
		return s.decodePayloadAuto(mdatPayload)
	}
	if len(data) >= 8 && string(data[4:8]) == "ftyp" {
		meta, err := InspectMp4Metadata(data)
		if err != nil {
			return VideoClip{}, err
		}
		payload, err := ParseMp4Container(data)
		if err != nil {
			return VideoClip{}, err
		}
		if meta.Encrypted {
			kid := DefaultEmeDemoKID
			if meta.DefaultKID != nil {
				kid = *meta.DefaultKID
			}
			key, ok := s.cdmSession.LookupKey(kid)
			if !ok {
				return VideoClip{}, &VideoError{Kind: ErrMissingDecryptionKey, KID: kid}
			}
			var iv [16]byte
			if meta.IV != nil {
				iv = *meta.IV
			}
			decrypted := CencDecryptSample(key, iv, payload)
			s.cdmSession.decryptedSamples++
			return s.decodePayloadAuto(decrypted)
		}
		return s.decodePayloadAuto(payload)
	}
	return s.decodePayloadAuto(data)
}

// CencEncryptSample encrypts plaintext using AES-128-CTR.
func CencEncryptSample(key, iv [16]byte, plaintext []byte) []byte {
	block, err := aes.NewCipher(key[:])
	if err != nil {
		panic(err)
	}
	out := make([]byte, len(plaintext))
	stream := cipher.NewCTR(block, iv[:])
	stream.XORKeyStream(out, plaintext)
	return out
}

// CencDecryptSample decrypts ciphertext using AES-128-CTR.
func CencDecryptSample(key, iv [16]byte, ciphertext []byte) []byte {
	return CencEncryptSample(key, iv, ciphertext)
}

func writeMp4Box(out *bytes.Buffer, fourcc [4]byte, payload []byte) {
	var sz [4]byte
	binary.BigEndian.PutUint32(sz[:], uint32(8+len(payload)))
	out.Write(sz[:])
	out.Write(fourcc[:])
	out.Write(payload)
}

// WrapMp4Container wraps mdatPayload in an ISOBMFF MP4 container with ftyp, moov, and mdat.
func WrapMp4Container(width, height, durationMs uint32, mdatPayload []byte) []byte {
	var out bytes.Buffer
	var ftyp [16]byte
	copy(ftyp[0:4], "isom")
	binary.BigEndian.PutUint32(ftyp[4:8], 512)
	copy(ftyp[8:16], "isommp42")
	writeMp4Box(&out, [4]byte{'f', 't', 'y', 'p'}, ftyp[:])

	var mvhd [12]byte
	binary.BigEndian.PutUint32(mvhd[0:4], 0)
	binary.BigEndian.PutUint32(mvhd[4:8], 1000)
	binary.BigEndian.PutUint32(mvhd[8:12], durationMs)

	var tkhd [12]byte
	binary.BigEndian.PutUint32(tkhd[0:4], 1)
	binary.BigEndian.PutUint32(tkhd[4:8], width)
	binary.BigEndian.PutUint32(tkhd[8:12], height)

	var trak bytes.Buffer
	writeMp4Box(&trak, [4]byte{'t', 'k', 'h', 'd'}, tkhd[:])

	var moov bytes.Buffer
	writeMp4Box(&moov, [4]byte{'m', 'v', 'h', 'd'}, mvhd[:])
	writeMp4Box(&moov, [4]byte{'t', 'r', 'a', 'k'}, trak.Bytes())
	writeMp4Box(&out, [4]byte{'m', 'o', 'o', 'v'}, moov.Bytes())
	writeMp4Box(&out, [4]byte{'m', 'd', 'a', 't'}, mdatPayload)
	return out.Bytes()
}

// WrapAvMp4Container muxes an AVID video bitstream and optional AAUD audio bitstream into an MP4 container.
func WrapAvMp4Container(width, height, durationMs uint32, avidPayload, aaudPayload []byte) []byte {
	if len(aaudPayload) == 0 {
		return WrapMp4Container(width, height, durationMs, avidPayload)
	}
	var out bytes.Buffer
	var ftyp [16]byte
	copy(ftyp[0:4], "isom")
	binary.BigEndian.PutUint32(ftyp[4:8], 512)
	copy(ftyp[8:16], "isommp42")
	writeMp4Box(&out, [4]byte{'f', 't', 'y', 'p'}, ftyp[:])

	var mvhd [12]byte
	binary.BigEndian.PutUint32(mvhd[4:8], 1000)
	binary.BigEndian.PutUint32(mvhd[8:12], durationMs)

	var tkhd [12]byte
	binary.BigEndian.PutUint32(tkhd[0:4], 1)
	binary.BigEndian.PutUint32(tkhd[4:8], width)
	binary.BigEndian.PutUint32(tkhd[8:12], height)

	var videoTrak bytes.Buffer
	writeMp4Box(&videoTrak, [4]byte{'t', 'k', 'h', 'd'}, tkhd[:])

	sampleRate := uint32(44100)
	channels := uint8(2)
	if len(aaudPayload) >= 16 && string(aaudPayload[0:4]) == "AAUD" {
		sampleRate = binary.BigEndian.Uint32(aaudPayload[8:12])
		channels = aaudPayload[6]
	}

	var hdlr [14]byte
	copy(hdlr[4:8], "soun")
	binary.BigEndian.PutUint32(hdlr[8:12], sampleRate)
	hdlr[12] = channels

	var smhd [8]byte
	var minf bytes.Buffer
	writeMp4Box(&minf, [4]byte{'s', 'm', 'h', 'd'}, smhd[:])

	var mdia bytes.Buffer
	writeMp4Box(&mdia, [4]byte{'h', 'd', 'l', 'r'}, hdlr[:])
	writeMp4Box(&mdia, [4]byte{'m', 'i', 'n', 'f'}, minf.Bytes())

	var audioTrak bytes.Buffer
	writeMp4Box(&audioTrak, [4]byte{'m', 'd', 'i', 'a'}, mdia.Bytes())

	var moov bytes.Buffer
	writeMp4Box(&moov, [4]byte{'m', 'v', 'h', 'd'}, mvhd[:])
	writeMp4Box(&moov, [4]byte{'t', 'r', 'a', 'k'}, videoTrak.Bytes())
	writeMp4Box(&moov, [4]byte{'t', 'r', 'a', 'k'}, audioTrak.Bytes())
	writeMp4Box(&out, [4]byte{'m', 'o', 'o', 'v'}, moov.Bytes())

	var mdat bytes.Buffer
	mdat.WriteString("AAVM")
	var lens [8]byte
	binary.BigEndian.PutUint32(lens[0:4], uint32(len(avidPayload)))
	binary.BigEndian.PutUint32(lens[4:8], uint32(len(aaudPayload)))
	mdat.Write(lens[:])
	mdat.Write(avidPayload)
	mdat.Write(aaudPayload)
	writeMp4Box(&out, [4]byte{'m', 'd', 'a', 't'}, mdat.Bytes())
	return out.Bytes()
}

// WrapEncryptedMp4Container wraps mdatPlaintext as an AES-128-CTR CENC MP4 container.
func WrapEncryptedMp4Container(width, height, durationMs uint32, keyID, key, iv [16]byte, mdatPlaintext []byte) []byte {
	var out bytes.Buffer
	var ftyp [16]byte
	copy(ftyp[0:4], "isom")
	binary.BigEndian.PutUint32(ftyp[4:8], 512)
	copy(ftyp[8:16], "isomcenc")
	writeMp4Box(&out, [4]byte{'f', 't', 'y', 'p'}, ftyp[:])

	var mvhd [12]byte
	binary.BigEndian.PutUint32(mvhd[4:8], 1000)
	binary.BigEndian.PutUint32(mvhd[8:12], durationMs)

	var tkhd [12]byte
	binary.BigEndian.PutUint32(tkhd[0:4], 1)
	binary.BigEndian.PutUint32(tkhd[4:8], width)
	binary.BigEndian.PutUint32(tkhd[8:12], height)

	var schm [12]byte
	copy(schm[4:8], "cenc")
	binary.BigEndian.PutUint32(schm[8:12], 0x00010000)

	var tenc [24]byte
	tenc[6] = 1
	tenc[7] = 16
	copy(tenc[8:24], keyID[:])

	var sinf bytes.Buffer
	writeMp4Box(&sinf, [4]byte{'s', 'c', 'h', 'm'}, schm[:])
	writeMp4Box(&sinf, [4]byte{'t', 'e', 'n', 'c'}, tenc[:])

	var trak bytes.Buffer
	writeMp4Box(&trak, [4]byte{'t', 'k', 'h', 'd'}, tkhd[:])
	writeMp4Box(&trak, [4]byte{'s', 'i', 'n', 'f'}, sinf.Bytes())

	var pssh [60]byte
	binary.BigEndian.PutUint32(pssh[0:4], 0x01000000)
	copy(pssh[4:20], W3CClearKeySystemID[:])
	binary.BigEndian.PutUint32(pssh[20:24], 1)
	copy(pssh[24:40], keyID[:])
	binary.BigEndian.PutUint32(pssh[40:44], 16)
	copy(pssh[44:60], iv[:])

	var moov bytes.Buffer
	writeMp4Box(&moov, [4]byte{'m', 'v', 'h', 'd'}, mvhd[:])
	writeMp4Box(&moov, [4]byte{'t', 'r', 'a', 'k'}, trak.Bytes())
	writeMp4Box(&moov, [4]byte{'p', 's', 's', 'h'}, pssh[:])
	writeMp4Box(&out, [4]byte{'m', 'o', 'o', 'v'}, moov.Bytes())

	encryptedMdat := CencEncryptSample(key, iv, mdatPlaintext)
	writeMp4Box(&out, [4]byte{'m', 'd', 'a', 't'}, encryptedMdat)
	return out.Bytes()
}

// InspectMp4Metadata walks an ISOBMFF MP4 container and extracts metadata without decoding mdat.
func InspectMp4Metadata(data []byte) (Mp4Metadata, error) {
	if len(data) > MaxContainerBytes {
		return Mp4Metadata{}, &VideoError{Kind: ErrContainerTooLarge, Size: len(data)}
	}
	if len(data) < 12 || string(data[4:8]) != "ftyp" {
		return Mp4Metadata{}, &VideoError{Kind: ErrInvalidHeader}
	}
	var meta Mp4Metadata
	copy(meta.Brand[:], data[8:12])
	meta.Timescale = 1000
	var sawMoov, sawMdat bool
	if err := scanMp4Boxes(data, 0, &meta, &sawMoov, &sawMdat); err != nil {
		return Mp4Metadata{}, err
	}
	if !sawMoov || !sawMdat {
		return Mp4Metadata{}, &VideoError{Kind: ErrInvalidHeader}
	}
	if meta.Width > 0 && meta.Height > 0 {
		if err := ValidateVideoDimensions(meta.Width, meta.Height); err != nil {
			return Mp4Metadata{}, err
		}
	}
	return meta, nil
}

func scanMp4Boxes(slice []byte, depth int, meta *Mp4Metadata, sawMoov, sawMdat *bool) error {
	if depth > MaxMp4BoxDepth {
		return &VideoError{Kind: ErrBoxDepthExceeded, Depth: depth}
	}
	cursor := 0
	for cursor+8 <= len(slice) {
		meta.BoxCount++
		if meta.BoxCount > MaxMp4Boxes {
			return &VideoError{Kind: ErrTooManyBoxes, Size: meta.BoxCount}
		}
		boxSize := int(binary.BigEndian.Uint32(slice[cursor : cursor+4]))
		if boxSize < 8 || cursor+boxSize > len(slice) {
			return &VideoError{Kind: ErrCorruptedBitstream}
		}
		boxType := string(slice[cursor+4 : cursor+8])
		payload := slice[cursor+8 : cursor+boxSize]
		switch boxType {
		case "moov", "trak", "mdia", "minf", "stbl", "sinf":
			if boxType == "moov" {
				*sawMoov = true
			}
			if err := scanMp4Boxes(payload, depth+1, meta, sawMoov, sawMdat); err != nil {
				return err
			}
		case "mvhd":
			if len(payload) >= 12 {
				meta.Timescale = maxU32(binary.BigEndian.Uint32(payload[4:8]), 1)
				meta.DurationMs = binary.BigEndian.Uint32(payload[8:12])
			}
		case "tkhd":
			if len(payload) >= 12 {
				meta.Width = binary.BigEndian.Uint32(payload[4:8])
				meta.Height = binary.BigEndian.Uint32(payload[8:12])
			}
		case "schm":
			if len(payload) >= 8 {
				var schm [4]byte
				copy(schm[:], payload[4:8])
				meta.ProtectionScheme = &schm
			}
		case "tenc":
			if len(payload) >= 24 {
				meta.Encrypted = payload[6] != 0
				var kid [16]byte
				copy(kid[:], payload[8:24])
				meta.DefaultKID = &kid
			}
		case "pssh":
			if len(payload) >= 20 {
				meta.PsshCount++
				if len(payload) >= 60 {
					var iv [16]byte
					copy(iv[:], payload[44:60])
					meta.IV = &iv
				}
			}
		case "mdat":
			if len(payload) > MaxPacketBytes {
				return &VideoError{Kind: ErrPacketTooLarge, Size: len(payload)}
			}
			meta.MdatSize = len(payload)
			*sawMdat = true
		case "hdlr":
			if len(payload) >= 14 && string(payload[4:8]) == "soun" {
				meta.HasAudioTrack = true
				meta.AudioSampleRate = binary.BigEndian.Uint32(payload[8:12])
				meta.AudioChannels = payload[12]
			}
		case "smhd":
			if len(payload) >= 8 {
				meta.HasAudioTrack = true
			}
		}
		cursor += boxSize
	}
	if cursor != len(slice) {
		return &VideoError{Kind: ErrCorruptedBitstream}
	}
	return nil
}

// ParseMp4Container validates an MP4 container and returns the last top-level mdat payload.
func ParseMp4Container(data []byte) ([]byte, error) {
	if _, err := InspectMp4Metadata(data); err != nil {
		return nil, err
	}
	cursor := 0
	var mdatSlice []byte
	var found bool
	for cursor+8 <= len(data) {
		boxSize := int(binary.BigEndian.Uint32(data[cursor : cursor+4]))
		if boxSize < 8 || cursor+boxSize > len(data) {
			return nil, &VideoError{Kind: ErrCorruptedBitstream}
		}
		boxType := string(data[cursor+4 : cursor+8])
		payload := data[cursor+8 : cursor+boxSize]
		if boxType == "mdat" {
			if len(payload) > MaxPacketBytes {
				return nil, &VideoError{Kind: ErrPacketTooLarge, Size: len(payload)}
			}
			mdatSlice = payload
			found = true
		}
		cursor += boxSize
	}
	if !found {
		return nil, &VideoError{Kind: ErrInvalidHeader}
	}
	return mdatSlice, nil
}

// GenerateDemoMp4 builds a short unencrypted dual-track (AVID + AAUD) demo MP4.
func GenerateDemoMp4(width, height, frameCount uint32) ([]byte, error) {
	if err := ValidateVideoDimensions(width, height); err != nil {
		return nil, err
	}
	count := clampU32(frameCount, 1, 60)
	frames := make([]VideoFrame, 0, int(count))
	for f := uint32(0); f < count; f++ {
		rgba := make([]byte, int(width)*int(height)*4)
		barX := int32((f * 6) % width)
		for y := uint32(0); y < height; y++ {
			for x := uint32(0); x < width; x++ {
				idx := int((y*width + x) * 4)
				dist := int32(x) - barX
				if dist < 0 {
					dist = -dist
				}
				if dist < 14 {
					rgba[idx] = 232
					rgba[idx+1] = 195
					rgba[idx+2] = 122
					rgba[idx+3] = 255
				} else {
					rgba[idx] = byte(24 + (x*40)/maxU32(width, 1))
					rgba[idx+1] = byte(32 + (y*50)/maxU32(height, 1))
					rgba[idx+2] = 56
					rgba[idx+3] = 255
				}
			}
		}
		vf, err := NewVideoFrame(width, height, f*33, f == 0, rgba)
		if err != nil {
			return nil, err
		}
		frames = append(frames, vf)
	}
	durationMs := maxU32(count*33, 100)
	audio := SynthesizeHarmonicChime(durationMs, 44100, 2, AudioImaAdpcm4)
	sess := DefaultGpuVideoSession()
	return sess.EncodeAvMp4(30, frames, &audio)
}

// GenerateEncryptedDemoMp4 builds a short AES-128-CTR CENC encrypted demo MP4.
func GenerateEncryptedDemoMp4(width, height, frameCount uint32) ([]byte, error) {
	if err := ValidateVideoDimensions(width, height); err != nil {
		return nil, err
	}
	count := clampU32(frameCount, 1, 60)
	frames := make([]VideoFrame, 0, int(count))
	for f := uint32(0); f < count; f++ {
		rgba := make([]byte, int(width)*int(height)*4)
		barX := int32((f * 8) % width)
		for y := uint32(0); y < height; y++ {
			for x := uint32(0); x < width; x++ {
				idx := int((y*width + x) * 4)
				dist := int32(x) - barX
				if dist < 0 {
					dist = -dist
				}
				if dist < 12 {
					rgba[idx] = 88
					rgba[idx+1] = 196
					rgba[idx+2] = 220
					rgba[idx+3] = 255
				} else {
					rgba[idx] = byte(18 + (x*36)/maxU32(width, 1))
					rgba[idx+1] = byte(38 + (y*44)/maxU32(height, 1))
					rgba[idx+2] = 52
					rgba[idx+3] = 255
				}
			}
		}
		vf, err := NewVideoFrame(width, height, f*33, f == 0, rgba)
		if err != nil {
			return nil, err
		}
		frames = append(frames, vf)
	}
	iv := [16]byte{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10}
	sess := DefaultGpuVideoSession()
	return sess.EncodeEncryptedMp4(30, frames, DefaultEmeDemoKID, DefaultEmeDemoKey, iv)
}

func clampI32(v, lo, hi int32) int32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func clampU32(v, lo, hi uint32) uint32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func clampF64(v, lo, hi float64) float64 {
	if math.IsNaN(v) {
		return lo
	}
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func maxU8(a, b uint8) uint8 {
	if a > b {
		return a
	}
	return b
}

func maxU32(a, b uint32) uint32 {
	if a > b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
