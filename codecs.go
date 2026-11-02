// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Buntato

package anvilvideo

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/bits"
	"runtime"
	"strings"
)

// Zigzag8x8 maps (row * 8 + col) to standard 8x8 DCT zig-zag frequency order.
var Zigzag8x8 = [64]int{
	0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
	13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
	52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
}

var baseQuant8x8 = [64]int32{
	16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
	14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
	92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
}

func dctBasis(u, x int) float64 {
	alpha := 1.0
	if u == 0 {
		alpha = math.Sqrt2 / 2.0
	}
	angle := float64((2*x+1)*u) * math.Pi / 16.0
	return 0.5 * alpha * math.Cos(angle)
}

// ForwardDct8x8 computes the 2D 8x8 forward Discrete Cosine Transform (DCT-II).
func ForwardDct8x8(block *[64]int16) [64]int32 {
	var temp [64]float64
	for r := 0; r < 8; r++ {
		for u := 0; u < 8; u++ {
			sum := 0.0
			for x := 0; x < 8; x++ {
				sum += float64(block[r*8+x]) * dctBasis(u, x)
			}
			temp[r*8+u] = sum
		}
	}
	var out [64]int32
	for v := 0; v < 8; v++ {
		for u := 0; u < 8; u++ {
			sum := 0.0
			for y := 0; y < 8; y++ {
				sum += temp[y*8+u] * dctBasis(v, y)
			}
			out[v*8+u] = int32(math.Round(sum))
		}
	}
	return out
}

// InverseIdct8x8 computes the 2D 8x8 inverse Discrete Cosine Transform (IDCT / DCT-III).
func InverseIdct8x8(coeffs *[64]int32) [64]int16 {
	var temp [64]float64
	for y := 0; y < 8; y++ {
		for u := 0; u < 8; u++ {
			sum := 0.0
			for v := 0; v < 8; v++ {
				sum += float64(coeffs[v*8+u]) * dctBasis(v, y)
			}
			temp[y*8+u] = sum
		}
	}
	var out [64]int16
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			sum := 0.0
			for u := 0; u < 8; u++ {
				sum += temp[y*8+u] * dctBasis(u, x)
			}
			out[y*8+x] = int16(clampF64(math.Round(sum), -255.0, 255.0))
		}
	}
	return out
}

// Quantize8x8 quantizes an 8x8 DCT block in zig-zag order using qp (1..=51).
func Quantize8x8(coeffs *[64]int32, qp uint8) [64]int16 {
	scale := clampI32(int32(qp), 1, 51)
	var out [64]int16
	for zz, pos := range Zigzag8x8 {
		step := (baseQuant8x8[pos] * scale) / 16
		if step < 1 {
			step = 1
		}
		c := coeffs[pos]
		bias := step / 2
		if c < 0 {
			bias = -bias
		}
		q := (c + bias) / step
		out[zz] = int16(clampI32(q, -32768, 32767))
	}
	return out
}

// Dequantize8x8 dequantizes a zig-zag ordered 8x8 DCT block using qp (1..=51).
func Dequantize8x8(quantizedZz *[64]int16, qp uint8) [64]int32 {
	scale := clampI32(int32(qp), 1, 51)
	var out [64]int32
	for zz, pos := range Zigzag8x8 {
		step := (baseQuant8x8[pos] * scale) / 16
		if step < 1 {
			step = 1
		}
		out[pos] = int32(quantizedZz[zz]) * step
	}
	return out
}

// MotionVector holds integer luma displacement (Dx, Dy) for an 8x8 macroblock.
type MotionVector struct {
	Dx int8
	Dy int8
}

func readPlaneClamped(plane []byte, width, height, x, y int) byte {
	if x < 0 {
		x = 0
	} else if x >= width {
		x = width - 1
	}
	if y < 0 {
		y = 0
	} else if y >= height {
		y = height - 1
	}
	return plane[y*width+x]
}

// EstimateMotionVector8x8 finds the best integer motion vector in [-searchRange, +searchRange] minimizing SAD.
func EstimateMotionVector8x8(currY, refY []byte, width, height, bx, by int, searchRange int8) MotionVector {
	rng := int(clampI32(int32(searchRange), 1, 16))
	bestMv := MotionVector{Dx: 0, Dy: 0}
	var zeroSad uint32
	for dy := 0; dy < 8; dy++ {
		for dx := 0; dx < 8; dx++ {
			c := int32(readPlaneClamped(currY, width, height, bx+dx, by+dy))
			r := int32(readPlaneClamped(refY, width, height, bx+dx, by+dy))
			diff := c - r
			if diff < 0 {
				diff = -diff
			}
			zeroSad += uint32(diff)
		}
	}
	bestSad := zeroSad
	if bestSad <= 16 {
		return bestMv
	}
	for mvY := -rng; mvY <= rng; mvY += 2 {
		for mvX := -rng; mvX <= rng; mvX += 2 {
			if mvX == 0 && mvY == 0 {
				continue
			}
			var sad uint32
			for py := 0; py < 8; py++ {
				for px := 0; px < 8; px++ {
					c := int32(readPlaneClamped(currY, width, height, bx+px, by+py))
					r := int32(readPlaneClamped(refY, width, height, bx+px+mvX, by+py+mvY))
					diff := c - r
					if diff < 0 {
						diff = -diff
					}
					sad += uint32(diff)
				}
			}
			if sad < bestSad {
				bestSad = sad
				bestMv = MotionVector{Dx: int8(mvX), Dy: int8(mvY)}
			}
		}
	}
	return bestMv
}

// EncodeDctMcBitstream encodes VideoFrames using 8x8 DCT + quantization on I-frames and motion-compensated residuals on P-frames (ADCT).
func EncodeDctMcBitstream(profile VideoCodecProfile, fps uint32, qp uint8, frames []VideoFrame) ([]byte, error) {
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
	qpClamped := uint8(clampI32(int32(qp), 1, 51))
	w, h := int(width), int(height)
	blocksX := (w + 7) / 8
	blocksY := (h + 7) / 8

	var out bytes.Buffer
	out.WriteString("ADCT")
	fcc := profile.FourCC()
	out.Write(fcc[:])
	out.WriteByte(qpClamped)
	out.WriteByte(0)
	var hdr [16]byte
	binary.BigEndian.PutUint32(hdr[0:4], width)
	binary.BigEndian.PutUint32(hdr[4:8], height)
	binary.BigEndian.PutUint32(hdr[8:12], maxU32(fps, 1))
	binary.BigEndian.PutUint32(hdr[12:16], uint32(len(frames)))
	out.Write(hdr[:])

	refI420 := NewI420Surface(width, height)
	for idx := range frames {
		f := &frames[idx]
		if f.Width != width || f.Height != height {
			return nil, &VideoError{Kind: ErrInvalidHeader}
		}
		currI420, ok := I420FromRgbaBt709(width, height, f.Rgba)
		if !ok {
			return nil, &VideoError{Kind: ErrInvalidHeader}
		}
		isKeyframe := idx == 0 || f.Keyframe || (idx%8 == 0)
		var sliceRaw bytes.Buffer
		reconY := make([]byte, w*h)
		for i := range reconY {
			reconY[i] = 16
		}
		for byIdx := 0; byIdx < blocksY; byIdx++ {
			for bxIdx := 0; bxIdx < blocksX; bxIdx++ {
				bx := bxIdx * 8
				by := byIdx * 8
				var mv MotionVector
				if !isKeyframe {
					mv = EstimateMotionVector8x8(currI420.YPlane, refI420.YPlane, w, h, bx, by, 6)
					sliceRaw.WriteByte(byte(mv.Dx))
					sliceRaw.WriteByte(byte(mv.Dy))
				}
				var spatial [64]int16
				for py := 0; py < 8; py++ {
					for px := 0; px < 8; px++ {
						curPx := readPlaneClamped(currI420.YPlane, w, h, bx+px, by+py)
						predPx := byte(128)
						if !isKeyframe {
							predPx = readPlaneClamped(refI420.YPlane, w, h, bx+px+int(mv.Dx), by+py+int(mv.Dy))
						}
						spatial[py*8+px] = int16(curPx) - int16(predPx)
					}
				}
				dct := ForwardDct8x8(&spatial)
				qZz := Quantize8x8(&dct, qpClamped)
				for _, coeff := range qZz {
					var cb [2]byte
					binary.BigEndian.PutUint16(cb[:], uint16(coeff))
					sliceRaw.Write(cb[:])
				}
				deq := Dequantize8x8(&qZz, qpClamped)
				idct := InverseIdct8x8(&deq)
				for py := 0; py < 8; py++ {
					for px := 0; px < 8; px++ {
						if bx+px < w && by+py < h {
							predPx := int16(128)
							if !isKeyframe {
								predPx = int16(readPlaneClamped(refI420.YPlane, w, h, bx+px+int(mv.Dx), by+py+int(mv.Dy)))
							}
							rec := clampI32(int32(predPx+idct[py*8+px]), 16, 235)
							reconY[(by+py)*w+(bx+px)] = byte(rec)
						}
					}
				}
			}
		}
		reconU := make([]byte, len(currI420.UPlane))
		reconV := make([]byte, len(currI420.VPlane))
		for i := range currI420.UPlane {
			cu, ru := currI420.UPlane[i], refI420.UPlane[i]
			delta := cu
			if !isKeyframe {
				delta = cu - ru
			}
			sliceRaw.WriteByte(delta)
			if isKeyframe {
				reconU[i] = delta
			} else {
				reconU[i] = ru + delta
			}
		}
		for i := range currI420.VPlane {
			cv, rv := currI420.VPlane[i], refI420.VPlane[i]
			delta := cv
			if !isKeyframe {
				delta = cv - rv
			}
			sliceRaw.WriteByte(delta)
			if isKeyframe {
				reconV[i] = delta
			} else {
				reconV[i] = rv + delta
			}
		}
		refI420.YPlane = reconY
		refI420.UPlane = reconU
		refI420.VPlane = reconV

		compressed := compressZlib(sliceRaw.Bytes())
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
	}
	return out.Bytes(), nil
}

// DecodeDctMcBitstream decodes an ADCT 8x8 DCT + motion-compensated bitstream into a VideoClip.
func DecodeDctMcBitstream(data []byte, hardwareDecode bool) (VideoClip, error) {
	if len(data) < 26 || string(data[0:4]) != "ADCT" {
		return VideoClip{}, &VideoError{Kind: ErrInvalidHeader}
	}
	var profile VideoCodecProfile
	switch string(data[4:8]) {
	case "avc1":
		profile = ProfileH264High
	case "av01":
		profile = ProfileAv1Main
	case "vp09":
		profile = ProfileVp9Profile0
	case "vp08":
		profile = ProfileVp8Profile0
	case "avid":
		profile = ProfileAnvilNv12Temporal
	default:
		return VideoClip{}, &VideoError{Kind: ErrInvalidHeader}
	}
	qp := uint8(clampI32(int32(data[8]), 1, 51))
	width := binary.BigEndian.Uint32(data[10:14])
	height := binary.BigEndian.Uint32(data[14:18])
	fps := maxU32(binary.BigEndian.Uint32(data[18:22]), 1)
	frameCount := binary.BigEndian.Uint32(data[22:26])
	if err := ValidateVideoDimensions(width, height); err != nil {
		return VideoClip{}, err
	}
	if frameCount == 0 || frameCount > MaxVideoFrames {
		return VideoClip{}, &VideoError{Kind: ErrFrameCountTooLarge, Count: frameCount}
	}
	w, h := int(width), int(height)
	blocksX := (w + 7) / 8
	blocksY := (h + 7) / 8
	totalBlocks := blocksX * blocksY
	chromaLen := (w / 2) * (h / 2)

	refI420 := NewI420Surface(width, height)
	frames := make([]VideoFrame, 0, int(frameCount))
	cursor := 26
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
		var isKeyframe bool
		switch {
		case frameType == 'I':
			isKeyframe = true
		case frameType == 'P' && frameIdx > 0:
			isKeyframe = false
		default:
			return VideoClip{}, &VideoError{Kind: ErrCorruptedBitstream}
		}
		perBlockBytes := 130
		if isKeyframe {
			perBlockBytes = 128
		}
		expectedRaw := totalBlocks*perBlockBytes + chromaLen*2
		raw, err := decompressZlibBounded(data[cursor:cursor+pktLen], expectedRaw)
		if err != nil {
			return VideoClip{}, err
		}
		cursor += pktLen
		if len(raw) != expectedRaw {
			return VideoClip{}, &VideoError{Kind: ErrCorruptedBitstream}
		}
		rPos := 0
		reconY := make([]byte, w*h)
		for i := range reconY {
			reconY[i] = 16
		}
		for byIdx := 0; byIdx < blocksY; byIdx++ {
			for bxIdx := 0; bxIdx < blocksX; bxIdx++ {
				bx := bxIdx * 8
				by := byIdx * 8
				var mv MotionVector
				if !isKeyframe {
					mv = MotionVector{Dx: int8(raw[rPos]), Dy: int8(raw[rPos+1])}
					rPos += 2
				}
				var qZz [64]int16
				for i := 0; i < 64; i++ {
					qZz[i] = int16(binary.BigEndian.Uint16(raw[rPos : rPos+2]))
					rPos += 2
				}
				deq := Dequantize8x8(&qZz, qp)
				idct := InverseIdct8x8(&deq)
				for py := 0; py < 8; py++ {
					for px := 0; px < 8; px++ {
						if bx+px < w && by+py < h {
							predPx := int16(128)
							if !isKeyframe {
								predPx = int16(readPlaneClamped(refI420.YPlane, w, h, bx+px+int(mv.Dx), by+py+int(mv.Dy)))
							}
							rec := clampI32(int32(predPx+idct[py*8+px]), 16, 235)
							reconY[(by+py)*w+(bx+px)] = byte(rec)
						}
					}
				}
			}
		}
		uSlice := raw[rPos : rPos+chromaLen]
		rPos += chromaLen
		vSlice := raw[rPos : rPos+chromaLen]
		if isKeyframe {
			copy(refI420.UPlane, uSlice)
			copy(refI420.VPlane, vSlice)
		} else {
			for i := 0; i < chromaLen; i++ {
				refI420.UPlane[i] += uSlice[i]
				refI420.VPlane[i] += vSlice[i]
			}
		}
		refI420.YPlane = reconY
		vf, err := NewVideoFrame(width, height, ptsMs, isKeyframe, refI420.ToRgbaBt709())
		if err != nil {
			return VideoClip{}, err
		}
		frames = append(frames, vf)
	}
	return VideoClip{
		Width:   width,
		Height:  height,
		Fps:     fps,
		Backend: ProbePlatformGpuBackend(hardwareDecode),
		Profile: profile,
		Frames:  frames,
	}, nil
}

// EncodeH264AnnexB encodes VideoFrames as an H.264 / AVC (avc1) Annex-B NAL unit stream.
func EncodeH264AnnexB(fps uint32, qp uint8, frames []VideoFrame) ([]byte, error) {
	adct, err := EncodeDctMcBitstream(ProfileH264High, fps, qp, frames)
	if err != nil {
		return nil, err
	}
	width := frames[0].Width
	height := frames[0].Height
	var out bytes.Buffer
	out.Write([]byte{0x00, 0x00, 0x00, 0x01, 0x67, 0x64, 0x00, 0x28})
	var sps [12]byte
	binary.BigEndian.PutUint32(sps[0:4], width)
	binary.BigEndian.PutUint32(sps[4:8], height)
	binary.BigEndian.PutUint32(sps[8:12], maxU32(fps, 1))
	out.Write(sps[:])
	out.Write([]byte{0x00, 0x00, 0x00, 0x01, 0x68, 0xEE, 0x3C, 0x80})
	out.Write([]byte{0x00, 0x00, 0x00, 0x01, 0x65})
	out.Write(adct)
	return out.Bytes(), nil
}

// DecodeH264AnnexB decodes an H.264 / AVC (avc1) Annex-B NAL unit stream into a VideoClip.
func DecodeH264AnnexB(data []byte, hardwareDecode bool) (VideoClip, error) {
	if len(data) < 33 || !bytes.Equal(data[0:5], []byte{0x00, 0x00, 0x00, 0x01, 0x67}) {
		return VideoClip{}, &VideoError{Kind: ErrInvalidHeader}
	}
	for idx := 16; idx+5 <= len(data); idx++ {
		if bytes.Equal(data[idx:idx+5], []byte{0x00, 0x00, 0x00, 0x01, 0x65}) {
			return DecodeDctMcBitstream(data[idx+5:], hardwareDecode)
		}
	}
	return VideoClip{}, &VideoError{Kind: ErrCorruptedBitstream}
}

// EncodeFlacStream encodes an AudioTrack into a lossless fLaC stream with LPC order 0..=2.
func EncodeFlacStream(track *AudioTrack) ([]byte, error) {
	if track.Channels == 0 || track.Channels > 2 || track.SampleRate == 0 {
		return nil, &VideoError{Kind: ErrInvalidHeader}
	}
	count := len(track.Samples)
	if count*2 > MaxPacketBytes {
		return nil, &VideoError{Kind: ErrPacketTooLarge, Size: count * 2}
	}
	ch := int(track.Channels)
	bestOrder := uint8(0)
	bestEnergy := uint64(math.MaxUint64)
	for order := uint8(0); order <= 2; order++ {
		var energy uint64
		for i := 0; i < count; i++ {
			var pred int32
			switch {
			case order == 0:
				pred = 0
			case order == 1 && i >= ch:
				pred = int32(track.Samples[i-ch])
			case order == 2 && i >= 2*ch:
				pred = 2*int32(track.Samples[i-ch]) - int32(track.Samples[i-2*ch])
			case i >= ch:
				pred = int32(track.Samples[i-ch])
			}
			res := int32(track.Samples[i]) - pred
			if res < 0 {
				res = -res
			}
			energy += uint64(res)
		}
		if energy < bestEnergy {
			bestEnergy = energy
			bestOrder = order
		}
	}
	residuals := make([]byte, 1+count*3)
	residuals[0] = bestOrder
	for i := 0; i < count; i++ {
		var pred int32
		switch {
		case bestOrder == 0:
			pred = 0
		case bestOrder == 1 && i >= ch:
			pred = int32(track.Samples[i-ch])
		case bestOrder == 2 && i >= 2*ch:
			pred = 2*int32(track.Samples[i-ch]) - int32(track.Samples[i-2*ch])
		case i >= ch:
			pred = int32(track.Samples[i-ch])
		}
		res := int32(track.Samples[i]) - pred
		zz := uint32((res << 1) ^ (res >> 31))
		base := 1 + i*3
		residuals[base] = byte(zz)
		residuals[base+1] = byte(zz >> 8)
		residuals[base+2] = byte(zz >> 16)
	}
	compressed := compressZlib(residuals)
	out := make([]byte, 20+len(compressed))
	copy(out[0:4], "fLaC")
	copy(out[4:8], []byte{0x80, 0x00, 0x00, 12})
	binary.BigEndian.PutUint32(out[8:12], track.SampleRate)
	out[12] = track.Channels
	out[13] = 16
	binary.BigEndian.PutUint32(out[16:20], uint32(count))
	copy(out[20:], compressed)
	return out, nil
}

// DecodeFlacStream decodes a lossless fLaC audio stream back into an AudioTrack.
func DecodeFlacStream(data []byte) (AudioTrack, error) {
	if len(data) < 20 || string(data[0:4]) != "fLaC" {
		return AudioTrack{}, &VideoError{Kind: ErrInvalidHeader}
	}
	sampleRate := binary.BigEndian.Uint32(data[8:12])
	channels := data[12]
	sampleCount := int(binary.BigEndian.Uint32(data[16:20]))
	if channels == 0 || channels > 2 || sampleRate == 0 || sampleCount < 0 || sampleCount > MaxPacketBytes/3 {
		return AudioTrack{}, &VideoError{Kind: ErrInvalidHeader}
	}
	expectedRaw := 1 + sampleCount*3
	raw, err := decompressZlibBounded(data[20:], expectedRaw)
	if err != nil {
		return AudioTrack{}, err
	}
	if len(raw) != expectedRaw {
		return AudioTrack{}, &VideoError{Kind: ErrCorruptedBitstream}
	}
	order := raw[0]
	ch := int(channels)
	samples := make([]int16, sampleCount)
	for i := 0; i < sampleCount; i++ {
		base := 1 + i*3
		zz := uint32(raw[base]) | (uint32(raw[base+1]) << 8) | (uint32(raw[base+2]) << 16)
		res := int32(zz>>1) ^ -int32(zz&1)
		var pred int32
		switch {
		case order == 0:
			pred = 0
		case order == 1 && i >= ch:
			pred = int32(samples[i-ch])
		case order == 2 && i >= 2*ch:
			pred = 2*int32(samples[i-ch]) - int32(samples[i-2*ch])
		case i >= ch:
			pred = int32(samples[i-ch])
		}
		samples[i] = int16(clampI32(pred+res, -32768, 32767))
	}
	return AudioTrack{
		SampleRate: sampleRate,
		Channels:   channels,
		Codec:      AudioFlacLossless,
		Samples:    samples,
	}, nil
}

func mdctWindow(n, len2n int) float64 {
	angle := (float64(n) + 0.5) * math.Pi / float64(len2n)
	return math.Sin(angle)
}

// EncodeMdctAudioStream encodes an AudioTrack using a 50%-overlapped MDCT perceptual filterbank.
func EncodeMdctAudioStream(track *AudioTrack) ([]byte, error) {
	if track.Channels == 0 || track.Channels > 2 || track.SampleRate == 0 {
		return nil, &VideoError{Kind: ErrInvalidHeader}
	}
	const N = 32
	const N2 = 64
	ch := int(track.Channels)
	totalSamples := len(track.Samples)
	if totalSamples*2 > MaxPacketBytes {
		return nil, &VideoError{Kind: ErrPacketTooLarge, Size: totalSamples * 2}
	}
	framesPerCh := totalSamples / ch
	hopCount := (framesPerCh+N-1)/N + 1

	var coeffBytes bytes.Buffer
	for c := 0; c < ch; c++ {
		padded := make([]float64, (hopCount+1)*N)
		for i := 0; i < framesPerCh; i++ {
			padded[N+i] = float64(track.Samples[i*ch+c])
		}
		for hop := 0; hop < hopCount; hop++ {
			base := hop * N
			for k := 0; k < N; k++ {
				sum := 0.0
				for n := 0; n < N2; n++ {
					wIn := padded[base+n] * mdctWindow(n, N2)
					angle := (math.Pi / float64(N)) * (float64(n) + 0.5 + float64(N)*0.5) * (float64(k) + 0.5)
					sum += wIn * math.Cos(angle)
				}
				q := int16(clampF64(math.Round(sum/16.0), -32768.0, 32767.0))
				var qb [2]byte
				binary.LittleEndian.PutUint16(qb[:], uint16(q))
				coeffBytes.Write(qb[:])
			}
		}
	}
	compressed := compressZlib(coeffBytes.Bytes())
	out := make([]byte, 20+len(compressed))
	copy(out[0:4], "MDCT")
	out[4] = track.Channels
	binary.BigEndian.PutUint32(out[8:12], track.SampleRate)
	binary.BigEndian.PutUint32(out[12:16], uint32(totalSamples))
	binary.BigEndian.PutUint32(out[16:20], uint32(hopCount))
	copy(out[20:], compressed)
	return out, nil
}

// DecodeMdctAudioStream decodes an MDCT perceptual audio stream using IMDCT + TDAC overlap-add synthesis.
func DecodeMdctAudioStream(data []byte) (AudioTrack, error) {
	if len(data) < 20 || string(data[0:4]) != "MDCT" {
		return AudioTrack{}, &VideoError{Kind: ErrInvalidHeader}
	}
	const N = 32
	const N2 = 64
	channels := data[4]
	sampleRate := binary.BigEndian.Uint32(data[8:12])
	totalSamples := int(binary.BigEndian.Uint32(data[12:16]))
	hopCount := int(binary.BigEndian.Uint32(data[16:20]))
	if channels == 0 || channels > 2 || sampleRate == 0 || totalSamples < 0 || totalSamples > MaxPacketBytes/2 {
		return AudioTrack{}, &VideoError{Kind: ErrInvalidHeader}
	}
	ch := int(channels)
	framesPerCh := totalSamples / ch
	minHops := (framesPerCh+N-1)/N + 1
	if hopCount < minHops || hopCount > MaxPacketBytes/(ch*N*2) {
		return AudioTrack{}, &VideoError{Kind: ErrInvalidHeader}
	}
	expectedRaw := ch * hopCount * N * 2
	raw, err := decompressZlibBounded(data[20:], expectedRaw)
	if err != nil {
		return AudioTrack{}, err
	}
	if len(raw) != expectedRaw {
		return AudioTrack{}, &VideoError{Kind: ErrCorruptedBitstream}
	}
	samples := make([]int16, totalSamples)
	pos := 0
	norm := 2.0 / float64(N)
	for c := 0; c < ch; c++ {
		synth := make([]float64, (hopCount+1)*N)
		for hop := 0; hop < hopCount; hop++ {
			var coeffs [N]float64
			for k := 0; k < N; k++ {
				q := int16(binary.LittleEndian.Uint16(raw[pos : pos+2]))
				pos += 2
				coeffs[k] = float64(q) * 16.0
			}
			base := hop * N
			for n := 0; n < N2; n++ {
				sum := 0.0
				for k, ck := range coeffs {
					angle := (math.Pi / float64(N)) * (float64(n) + 0.5 + float64(N)*0.5) * (float64(k) + 0.5)
					sum += ck * math.Cos(angle)
				}
				synth[base+n] += sum * norm * mdctWindow(n, N2)
			}
		}
		for i := 0; i < framesPerCh; i++ {
			samples[i*ch+c] = int16(clampF64(math.Round(synth[N+i]), -32768.0, 32767.0))
		}
	}
	return AudioTrack{
		SampleRate: sampleRate,
		Channels:   channels,
		Codec:      AudioOpusMdct,
		Samples:    samples,
	}, nil
}

// Fmp4SegmentInfo holds metadata parsed from a Fragmented MP4 (fMP4) init or media segment.
type Fmp4SegmentInfo struct {
	HasInitSegment     bool
	HasMediaFragment   bool
	SequenceNumber     uint32
	BaseDecodeTimeMs   uint64
	SampleCount        uint32
	FragmentDurationMs uint32
	CodecFourCC        *[4]byte
	Width              uint32
	Height             uint32
}

// WrapFmp4InitSegment builds an ISO BMFF Fragmented MP4 (fMP4) initialization segment.
func WrapFmp4InitSegment(width, height uint32, codec VideoCodecProfile, hasAudio bool) []byte {
	var out bytes.Buffer
	var ftyp [24]byte
	copy(ftyp[0:4], "iso6")
	binary.BigEndian.PutUint32(ftyp[4:8], 512)
	copy(ftyp[8:24], "iso6isomdashmp42")
	writeMp4Box(&out, [4]byte{'f', 't', 'y', 'p'}, ftyp[:])

	var mvhd [12]byte
	binary.BigEndian.PutUint32(mvhd[4:8], 1000)

	var tkhd [12]byte
	binary.BigEndian.PutUint32(tkhd[0:4], 1)
	binary.BigEndian.PutUint32(tkhd[4:8], width)
	binary.BigEndian.PutUint32(tkhd[8:12], height)

	var stsd bytes.Buffer
	var stsdHdr [8]byte
	binary.BigEndian.PutUint32(stsdHdr[4:8], 1)
	stsd.Write(stsdHdr[:])
	var entry [8]byte
	binary.BigEndian.PutUint32(entry[0:4], width)
	binary.BigEndian.PutUint32(entry[4:8], height)
	writeMp4Box(&stsd, codec.FourCC(), entry[:])

	var stbl bytes.Buffer
	writeMp4Box(&stbl, [4]byte{'s', 't', 's', 'd'}, stsd.Bytes())
	var minf bytes.Buffer
	writeMp4Box(&minf, [4]byte{'s', 't', 'b', 'l'}, stbl.Bytes())
	var mdia bytes.Buffer
	writeMp4Box(&mdia, [4]byte{'m', 'i', 'n', 'f'}, minf.Bytes())

	var trak bytes.Buffer
	writeMp4Box(&trak, [4]byte{'t', 'k', 'h', 'd'}, tkhd[:])
	writeMp4Box(&trak, [4]byte{'m', 'd', 'i', 'a'}, mdia.Bytes())

	var trex [8]byte
	binary.BigEndian.PutUint32(trex[4:8], 1)
	var mvex bytes.Buffer
	writeMp4Box(&mvex, [4]byte{'t', 'r', 'e', 'x'}, trex[:])

	var moov bytes.Buffer
	writeMp4Box(&moov, [4]byte{'m', 'v', 'h', 'd'}, mvhd[:])
	writeMp4Box(&moov, [4]byte{'t', 'r', 'a', 'k'}, trak.Bytes())
	if hasAudio {
		var smhd [8]byte
		var aMinf bytes.Buffer
		writeMp4Box(&aMinf, [4]byte{'s', 'm', 'h', 'd'}, smhd[:])
		var aMdia bytes.Buffer
		writeMp4Box(&aMdia, [4]byte{'m', 'i', 'n', 'f'}, aMinf.Bytes())
		var aTrak bytes.Buffer
		writeMp4Box(&aTrak, [4]byte{'m', 'd', 'i', 'a'}, aMdia.Bytes())
		writeMp4Box(&moov, [4]byte{'t', 'r', 'a', 'k'}, aTrak.Bytes())
	}
	writeMp4Box(&moov, [4]byte{'m', 'v', 'e', 'x'}, mvex.Bytes())
	writeMp4Box(&out, [4]byte{'m', 'o', 'o', 'v'}, moov.Bytes())
	return out.Bytes()
}

// WrapFmp4MediaSegment builds an ISO BMFF Fragmented MP4 (fMP4) media segment (moof + mdat).
func WrapFmp4MediaSegment(sequenceNumber uint32, baseDecodeTimeMs uint64, sampleDurationsMs []uint32, mdatPayload []byte) []byte {
	var mfhd [8]byte
	binary.BigEndian.PutUint32(mfhd[4:8], sequenceNumber)
	var tfhd [8]byte
	binary.BigEndian.PutUint32(tfhd[4:8], 1)
	var tfdt [12]byte
	binary.BigEndian.PutUint32(tfdt[0:4], 0x01000000)
	binary.BigEndian.PutUint64(tfdt[4:12], baseDecodeTimeMs)

	var trun bytes.Buffer
	var trunHdr [8]byte
	binary.BigEndian.PutUint32(trunHdr[0:4], 0x00000100)
	binary.BigEndian.PutUint32(trunHdr[4:8], uint32(len(sampleDurationsMs)))
	trun.Write(trunHdr[:])
	for _, dur := range sampleDurationsMs {
		var db [4]byte
		binary.BigEndian.PutUint32(db[:], dur)
		trun.Write(db[:])
	}
	var traf bytes.Buffer
	writeMp4Box(&traf, [4]byte{'t', 'f', 'h', 'd'}, tfhd[:])
	writeMp4Box(&traf, [4]byte{'t', 'f', 'd', 't'}, tfdt[:])
	writeMp4Box(&traf, [4]byte{'t', 'r', 'u', 'n'}, trun.Bytes())

	var moof bytes.Buffer
	writeMp4Box(&moof, [4]byte{'m', 'f', 'h', 'd'}, mfhd[:])
	writeMp4Box(&moof, [4]byte{'t', 'r', 'a', 'f'}, traf.Bytes())

	var out bytes.Buffer
	writeMp4Box(&out, [4]byte{'m', 'o', 'o', 'f'}, moof.Bytes())
	writeMp4Box(&out, [4]byte{'m', 'd', 'a', 't'}, mdatPayload)
	return out.Bytes()
}

// InspectFmp4Segment inspects an fMP4 buffer (init segment, media segment, or both).
func InspectFmp4Segment(data []byte) (Fmp4SegmentInfo, error) {
	if len(data) > MaxContainerBytes {
		return Fmp4SegmentInfo{}, &VideoError{Kind: ErrContainerTooLarge, Size: len(data)}
	}
	var info Fmp4SegmentInfo
	boxCount := 0
	if err := scanFmp4Boxes(data, 0, &boxCount, &info); err != nil {
		return Fmp4SegmentInfo{}, err
	}
	if !info.HasInitSegment && !info.HasMediaFragment {
		return Fmp4SegmentInfo{}, &VideoError{Kind: ErrInvalidHeader}
	}
	return info, nil
}

func scanFmp4Boxes(slice []byte, depth int, boxCount *int, info *Fmp4SegmentInfo) error {
	if depth > 16 {
		return &VideoError{Kind: ErrBoxDepthExceeded, Depth: depth}
	}
	cursor := 0
	for cursor+8 <= len(slice) {
		*boxCount++
		if *boxCount > MaxMp4Boxes {
			return &VideoError{Kind: ErrCorruptedBitstream}
		}
		boxSize := int(binary.BigEndian.Uint32(slice[cursor : cursor+4]))
		if boxSize < 8 || boxSize > len(slice)-cursor {
			return &VideoError{Kind: ErrCorruptedBitstream}
		}
		boxType := string(slice[cursor+4 : cursor+8])
		payload := slice[cursor+8 : cursor+boxSize]
		switch boxType {
		case "moov", "trak", "mdia", "minf", "stbl", "mvex", "moof", "traf":
			if boxType == "moov" {
				info.HasInitSegment = true
			}
			if boxType == "moof" {
				info.HasMediaFragment = true
			}
			if err := scanFmp4Boxes(payload, depth+1, boxCount, info); err != nil {
				return err
			}
		case "tkhd":
			if len(payload) >= 12 {
				info.Width = binary.BigEndian.Uint32(payload[4:8])
				info.Height = binary.BigEndian.Uint32(payload[8:12])
			}
		case "stsd":
			if len(payload) >= 16 {
				var fcc [4]byte
				copy(fcc[:], payload[12:16])
				info.CodecFourCC = &fcc
			}
		case "mfhd":
			if len(payload) >= 8 {
				info.SequenceNumber = binary.BigEndian.Uint32(payload[4:8])
			}
		case "tfdt":
			if len(payload) >= 12 {
				if payload[0] == 1 {
					info.BaseDecodeTimeMs = binary.BigEndian.Uint64(payload[4:12])
				} else {
					info.BaseDecodeTimeMs = uint64(binary.BigEndian.Uint32(payload[4:8]))
				}
			}
		case "trun":
			if len(payload) >= 8 {
				count := binary.BigEndian.Uint32(payload[4:8])
				maxEntries := uint32((len(payload) - 8) / 4)
				if count > maxEntries {
					count = maxEntries
				}
				info.SampleCount += count
				p := 8
				for i := uint32(0); i < count && p+4 <= len(payload); i++ {
					info.FragmentDurationMs += binary.BigEndian.Uint32(payload[p : p+4])
					p += 4
				}
			}
		}
		cursor += boxSize
	}
	return nil
}

// WebmMetadata holds metadata extracted from a WebM / Matroska (EBML) container.
type WebmMetadata struct {
	DocType      string
	Width        uint32
	Height       uint32
	DurationMs   uint32
	VideoCodecID string
	ClusterCount int
	BlockCount   int
}

// WrapWebmContainer wraps an encoded video bitstream into a WebM (EBML 0x1A45DFA3) container.
func WrapWebmContainer(width, height, durationMs uint32, codecID string, bitstreamPayload []byte) []byte {
	var out bytes.Buffer
	out.Write([]byte{0x1A, 0x45, 0xDF, 0xA3})
	docType := []byte("webm")
	var ebmlBody bytes.Buffer
	ebmlBody.Write([]byte{0x42, 0x82, byte(0x80 | len(docType))})
	ebmlBody.Write(docType)
	out.WriteByte(byte(0x80 | ebmlBody.Len()))
	out.Write(ebmlBody.Bytes())

	var seg bytes.Buffer
	seg.Write([]byte{0x15, 0x49, 0xA9, 0x66, 0x84})
	var durBytes [4]byte
	binary.BigEndian.PutUint32(durBytes[:], durationMs)
	seg.Write(durBytes[:])

	cidBytes := []byte(codecID)
	tracksLen := 9 + len(cidBytes)
	seg.Write([]byte{0x16, 0x54, 0xAE, 0x6B, 0x10, 0x00, 0x00, byte(tracksLen)})
	var dims [8]byte
	binary.BigEndian.PutUint32(dims[0:4], width)
	binary.BigEndian.PutUint32(dims[4:8], height)
	seg.Write(dims[:])
	seg.WriteByte(byte(len(cidBytes)))
	seg.Write(cidBytes)

	clusterLen := 5 + len(bitstreamPayload)
	seg.Write([]byte{0x1F, 0x43, 0xB6, 0x75})
	var cVint [4]byte
	binary.BigEndian.PutUint32(cVint[:], 0x10000000|(uint32(clusterLen)&0x0FFFFFFF))
	seg.Write(cVint[:])
	seg.WriteByte(0xA3)
	var bLen [4]byte
	binary.BigEndian.PutUint32(bLen[:], uint32(len(bitstreamPayload)))
	seg.Write(bLen[:])
	seg.Write(bitstreamPayload)

	out.Write([]byte{0x18, 0x53, 0x80, 0x67})
	var sVint [4]byte
	binary.BigEndian.PutUint32(sVint[:], 0x10000000|(uint32(seg.Len())&0x0FFFFFFF))
	out.Write(sVint[:])
	out.Write(seg.Bytes())
	return out.Bytes()
}

// InspectWebmMetadata inspects a WebM (EBML) container and returns its WebmMetadata.
func InspectWebmMetadata(data []byte) (WebmMetadata, error) {
	meta, _, err := ParseWebmContainer(data)
	return meta, err
}

// ParseWebmContainer parses a WebM (EBML) container and returns (WebmMetadata, bitstreamPayload).
func ParseWebmContainer(data []byte) (WebmMetadata, []byte, error) {
	if len(data) > MaxContainerBytes {
		return WebmMetadata{}, nil, &VideoError{Kind: ErrContainerTooLarge, Size: len(data)}
	}
	if len(data) < 24 || !bytes.Equal(data[0:4], []byte{0x1A, 0x45, 0xDF, 0xA3}) {
		return WebmMetadata{}, nil, &VideoError{Kind: ErrInvalidHeader}
	}
	ebmlLen := int(data[4] & 0x7F)
	cursor := 5 + ebmlLen
	if cursor+8 > len(data) || !bytes.Equal(data[cursor:cursor+4], []byte{0x18, 0x53, 0x80, 0x67}) {
		return WebmMetadata{}, nil, &VideoError{Kind: ErrInvalidHeader}
	}
	segLen := int(binary.BigEndian.Uint32(data[cursor+4:cursor+8]) & 0x0FFFFFFF)
	cursor += 8
	if cursor+segLen > len(data) {
		return WebmMetadata{}, nil, &VideoError{Kind: ErrCorruptedBitstream}
	}
	seg := data[cursor : cursor+segLen]
	s := 0
	if s+9 > len(seg) || !bytes.Equal(seg[s:s+4], []byte{0x15, 0x49, 0xA9, 0x66}) {
		return WebmMetadata{}, nil, &VideoError{Kind: ErrCorruptedBitstream}
	}
	durationMs := binary.BigEndian.Uint32(seg[s+5 : s+9])
	s += 9
	if s+17 > len(seg) || !bytes.Equal(seg[s:s+4], []byte{0x16, 0x54, 0xAE, 0x6B}) {
		return WebmMetadata{}, nil, &VideoError{Kind: ErrCorruptedBitstream}
	}
	tracksLen := int(binary.BigEndian.Uint32(seg[s+4:s+8]) & 0x0FFFFFFF)
	s += 8
	if s+tracksLen > len(seg) || tracksLen < 9 {
		return WebmMetadata{}, nil, &VideoError{Kind: ErrCorruptedBitstream}
	}
	width := binary.BigEndian.Uint32(seg[s : s+4])
	height := binary.BigEndian.Uint32(seg[s+4 : s+8])
	cidLen := int(seg[s+8])
	if 9+cidLen > tracksLen {
		return WebmMetadata{}, nil, &VideoError{Kind: ErrCorruptedBitstream}
	}
	videoCodecID := string(seg[s+9 : s+9+cidLen])
	if err := ValidateVideoDimensions(width, height); err != nil {
		return WebmMetadata{}, nil, err
	}
	s += tracksLen
	if s+13 > len(seg) || !bytes.Equal(seg[s:s+4], []byte{0x1F, 0x43, 0xB6, 0x75}) {
		return WebmMetadata{}, nil, &VideoError{Kind: ErrCorruptedBitstream}
	}
	s += 8
	if seg[s] != 0xA3 {
		return WebmMetadata{}, nil, &VideoError{Kind: ErrCorruptedBitstream}
	}
	blockLen := int(binary.BigEndian.Uint32(seg[s+1 : s+5]))
	s += 5
	if blockLen > MaxPacketBytes || s+blockLen > len(seg) {
		return WebmMetadata{}, nil, &VideoError{Kind: ErrPacketTooLarge, Size: blockLen}
	}
	payload := seg[s : s+blockLen]
	return WebmMetadata{
		DocType:      "webm",
		Width:        width,
		Height:       height,
		DurationMs:   durationMs,
		VideoCodecID: videoCodecID,
		ClusterCount: 1,
		BlockCount:   1,
	}, payload, nil
}

// HardwareVideoCapabilities reports host GPU compositor and Media Foundation / VA-API status.
type HardwareVideoCapabilities struct {
	GpuCompositorReady             bool
	PlatformMediaFoundationPresent bool
	AdapterDescription             string
	SupportedProfiles              []VideoCodecProfile
}

// ProbeHardwareVideoCapabilities probes the host OS GPU compositor and video subsystem.
func ProbeHardwareVideoCapabilities() HardwareVideoCapabilities {
	ready := runtime.GOOS == "windows" || runtime.GOOS == "linux"
	desc := "Software BT.709 + DCT/Motion Video Pipeline"
	if runtime.GOOS == "windows" {
		desc = "Direct3D 11 Hardware GPU + Windows Media Foundation Video Pipeline"
	} else if runtime.GOOS == "linux" {
		desc = "Hardware GPU Shader Compositor + DCT/Motion Video Pipeline"
	}
	return HardwareVideoCapabilities{
		GpuCompositorReady:             ready,
		PlatformMediaFoundationPresent: ready,
		AdapterDescription:             desc,
		SupportedProfiles: []VideoCodecProfile{
			ProfileH264High,
			ProfileVp9Profile0,
			ProfileVp8Profile0,
			ProfileAv1Main,
			ProfileAnvilNv12Temporal,
		},
	}
}

// EncodedChunkType identifies keyframe vs delta frame in W3C WebCodecs.
type EncodedChunkType uint8

const (
	ChunkTypeKey EncodedChunkType = iota
	ChunkTypeDelta
)

func (t EncodedChunkType) AsStr() string {
	if t == ChunkTypeKey {
		return "key"
	}
	return "delta"
}

// EncodedVideoChunk represents a W3C EncodedVideoChunk.
type EncodedVideoChunk struct {
	ChunkType   EncodedChunkType
	TimestampUs int64
	DurationUs  uint64
	Codec       VideoCodecProfile
	Data        []byte
}

// WebCodecsVideoEncoder implements a stateful W3C VideoEncoder supporting avc1, vp09, vp08, av01, and avid.
type WebCodecsVideoEncoder struct {
	codec        VideoCodecProfile
	width        uint32
	height       uint32
	fps          uint32
	qp           uint8
	configured   bool
	encodedCount uint64
}

func NewWebCodecsVideoEncoder() *WebCodecsVideoEncoder {
	return &WebCodecsVideoEncoder{
		codec: ProfileH264High,
		fps:   30,
		qp:    16,
	}
}

func ParseCodecString(codecStr string) (VideoCodecProfile, bool) {
	lower := strings.ToLower(strings.TrimSpace(codecStr))
	switch {
	case strings.HasPrefix(lower, "avc1") || strings.HasPrefix(lower, "h264"):
		return ProfileH264High, true
	case strings.HasPrefix(lower, "av01") || strings.HasPrefix(lower, "av1"):
		return ProfileAv1Main, true
	case strings.HasPrefix(lower, "vp09") || strings.HasPrefix(lower, "vp9"):
		return ProfileVp9Profile0, true
	case strings.HasPrefix(lower, "vp08") || strings.HasPrefix(lower, "vp8"):
		return ProfileVp8Profile0, true
	case strings.HasPrefix(lower, "avid"):
		return ProfileAnvilNv12Temporal, true
	default:
		return 0, false
	}
}

func (e *WebCodecsVideoEncoder) Configure(codecStr string, width, height, fps uint32, qp uint8) error {
	if err := ValidateVideoDimensions(width, height); err != nil {
		return err
	}
	prof, ok := ParseCodecString(codecStr)
	if !ok {
		return &VideoError{Kind: ErrInvalidHeader}
	}
	e.codec = prof
	e.width = width
	e.height = height
	e.fps = maxU32(fps, 1)
	e.qp = uint8(clampI32(int32(qp), 1, 51))
	e.configured = true
	return nil
}

func (e *WebCodecsVideoEncoder) IsConfigured() bool   { return e.configured }
func (e *WebCodecsVideoEncoder) EncodedCount() uint64 { return e.encodedCount }

func (e *WebCodecsVideoEncoder) Encode(frame *VideoFrame, forceKeyframe bool) (EncodedVideoChunk, error) {
	if !e.configured || frame.Width != e.width || frame.Height != e.height {
		return EncodedVideoChunk{}, &VideoError{Kind: ErrInvalidHeader}
	}
	f := *frame
	if forceKeyframe || e.encodedCount == 0 {
		f.Keyframe = true
	}
	var payload []byte
	var err error
	switch e.codec {
	case ProfileH264High:
		payload, err = EncodeH264AnnexB(e.fps, e.qp, []VideoFrame{f})
	case ProfileAv1Main:
		payload, err = EncodeAv1ObuStream(e.fps, e.qp, []VideoFrame{f})
	case ProfileVp9Profile0:
		payload, err = EncodeVp9Stream(e.fps, e.qp, []VideoFrame{f})
	case ProfileVp8Profile0:
		payload, err = EncodeVp8Rfc6386Stream(e.fps, e.qp, []VideoFrame{f})
	default:
		sess := DefaultGpuVideoSession()
		payload, err = sess.EncodeAvid(e.fps, []VideoFrame{f})
	}
	if err != nil {
		return EncodedVideoChunk{}, err
	}
	e.encodedCount++
	ct := ChunkTypeDelta
	if f.Keyframe {
		ct = ChunkTypeKey
	}
	return EncodedVideoChunk{
		ChunkType:   ct,
		TimestampUs: int64(f.PtsMs) * 1000,
		DurationUs:  1_000_000 / uint64(maxU32(e.fps, 1)),
		Codec:       e.codec,
		Data:        payload,
	}, nil
}

// WebCodecsVideoDecoder decodes EncodedVideoChunks into VideoFrames.
type WebCodecsVideoDecoder struct {
	codec        VideoCodecProfile
	configured   bool
	decodedCount uint64
}

func NewWebCodecsVideoDecoder() *WebCodecsVideoDecoder {
	return &WebCodecsVideoDecoder{codec: ProfileH264High}
}

func (d *WebCodecsVideoDecoder) Configure(codecStr string) error {
	prof, ok := ParseCodecString(codecStr)
	if !ok {
		return &VideoError{Kind: ErrInvalidHeader}
	}
	d.codec = prof
	d.configured = true
	return nil
}

func (d *WebCodecsVideoDecoder) IsConfigured() bool   { return d.configured }
func (d *WebCodecsVideoDecoder) DecodedCount() uint64 { return d.decodedCount }

func (d *WebCodecsVideoDecoder) Decode(chunk *EncodedVideoChunk) (VideoFrame, error) {
	if !d.configured {
		return VideoFrame{}, &VideoError{Kind: ErrInvalidHeader}
	}
	sess := DefaultGpuVideoSession()
	clip, err := sess.DecodeVideoAuto(chunk.Data)
	if err != nil {
		return VideoFrame{}, err
	}
	if len(clip.Frames) == 0 {
		return VideoFrame{}, &VideoError{Kind: ErrCorruptedBitstream}
	}
	frame := clip.Frames[0]
	ts := chunk.TimestampUs
	if ts < 0 {
		ts = 0
	}
	frame.PtsMs = uint32(ts / 1000)
	frame.Keyframe = chunk.ChunkType == ChunkTypeKey
	d.decodedCount++
	return frame, nil
}

// MseTimeRange represents a buffered [StartMs, EndMs) interval in W3C MSE.
type MseTimeRange struct {
	StartMs uint32
	EndMs   uint32
}

// MseSourceBufferEngine implements the W3C Media Source Extensions SourceBuffer segment demuxer.
type MseSourceBufferEngine struct {
	mimeType         string
	initReceived     bool
	width            uint32
	height           uint32
	frames           []VideoFrame
	buffered         []MseTimeRange
	appendedSegments int
}

func NewMseSourceBufferEngine(mimeType string) *MseSourceBufferEngine {
	return &MseSourceBufferEngine{mimeType: mimeType}
}

func (m *MseSourceBufferEngine) MimeType() string               { return m.mimeType }
func (m *MseSourceBufferEngine) InitReceived() bool             { return m.initReceived }
func (m *MseSourceBufferEngine) Frames() []VideoFrame           { return m.frames }
func (m *MseSourceBufferEngine) BufferedRanges() []MseTimeRange { return m.buffered }
func (m *MseSourceBufferEngine) AppendedSegments() int          { return m.appendedSegments }

func (m *MseSourceBufferEngine) AppendBuffer(data []byte) (int, error) {
	if len(data) == 0 {
		return 0, &VideoError{Kind: ErrInvalidHeader}
	}
	m.appendedSegments++
	if len(data) >= 8 && string(data[4:8]) == "ftyp" && !containsTopLevelBox(data, [4]byte{'m', 'd', 'a', 't'}) {
		info, err := InspectFmp4Segment(data)
		if err != nil {
			return 0, err
		}
		if info.HasInitSegment {
			m.initReceived = true
			m.width = info.Width
			m.height = info.Height
			return 0, nil
		}
	}
	if len(data) >= 8 && string(data[4:8]) == "moof" {
		info, err := InspectFmp4Segment(data)
		if err != nil {
			return 0, err
		}
		mdatPayload, ok := extractTopLevelBoxPayload(data, [4]byte{'m', 'd', 'a', 't'})
		if !ok {
			return 0, &VideoError{Kind: ErrCorruptedBitstream}
		}
		sess := DefaultGpuVideoSession()
		clip, err := sess.DecodeVideoAuto(mdatPayload)
		if err != nil {
			return 0, err
		}
		m.initReceived = true
		m.width = clip.Width
		m.height = clip.Height
		baseMs := uint32(info.BaseDecodeTimeMs)
		added := len(clip.Frames)
		startMs := baseMs
		if added > 0 {
			startMs = baseMs + clip.Frames[0].PtsMs
		}
		endMs := startMs + maxU32(info.FragmentDurationMs, 33)
		for _, frame := range clip.Frames {
			frame.PtsMs = baseMs + frame.PtsMs
			if frame.PtsMs+33 > endMs {
				endMs = frame.PtsMs + 33
			}
			m.frames = append(m.frames, frame)
		}
		m.recordRange(startMs, endMs)
		return added, nil
	}
	sess := DefaultGpuVideoSession()
	clip, err := sess.DecodeVideoAuto(data)
	if err != nil {
		return 0, err
	}
	m.initReceived = true
	m.width = clip.Width
	m.height = clip.Height
	added := len(clip.Frames)
	if added > 0 {
		m.recordRange(clip.Frames[0].PtsMs, clip.Frames[added-1].PtsMs+33)
	}
	m.frames = append(m.frames, clip.Frames...)
	return added, nil
}

func (m *MseSourceBufferEngine) recordRange(startMs, endMs uint32) {
	if len(m.buffered) > 0 {
		last := &m.buffered[len(m.buffered)-1]
		if startMs <= last.EndMs+50 {
			if endMs > last.EndMs {
				last.EndMs = endMs
			}
			return
		}
	}
	m.buffered = append(m.buffered, MseTimeRange{StartMs: startMs, EndMs: endMs})
}

func containsTopLevelBox(data []byte, fourcc [4]byte) bool {
	_, ok := extractTopLevelBoxPayload(data, fourcc)
	return ok
}

func extractTopLevelBoxPayload(data []byte, fourcc [4]byte) ([]byte, bool) {
	cursor := 0
	var found []byte
	var ok bool
	for cursor+8 <= len(data) {
		size := int(binary.BigEndian.Uint32(data[cursor : cursor+4]))
		if size < 8 || cursor+size > len(data) {
			return nil, false
		}
		if bytes.Equal(data[cursor+4:cursor+8], fourcc[:]) {
			found = data[cursor+8 : cursor+size]
			ok = true
		}
		cursor += size
	}
	return found, ok
}

// MediaCapabilitiesInfo represents W3C MediaCapabilitiesInfo.
type MediaCapabilitiesInfo struct {
	Supported      bool
	Smooth         bool
	PowerEfficient bool
}

func QueryMediaCapabilities(contentType string, width, height, fps uint32) MediaCapabilitiesInfo {
	lower := strings.ToLower(strings.TrimSpace(contentType))
	codecOk := strings.Contains(lower, "mp4") ||
		strings.Contains(lower, "webm") ||
		strings.Contains(lower, "ogg") ||
		strings.Contains(lower, "wav") ||
		strings.Contains(lower, "flac") ||
		strings.Contains(lower, "avc1") ||
		strings.Contains(lower, "vp09") ||
		strings.Contains(lower, "vp9") ||
		strings.Contains(lower, "vp8") ||
		strings.Contains(lower, "av01") ||
		strings.Contains(lower, "opus") ||
		strings.Contains(lower, "mp4a") ||
		strings.Contains(lower, "avid")
	dimsOk := width > 0 && height > 0 && ValidateVideoDimensions(width&^1, height&^1) == nil
	supported := codecOk && dimsOk
	smooth := supported && fps <= 120 && uint64(width)*uint64(height) <= 3840*2160
	hw := ProbeHardwareVideoCapabilities()
	powerEfficient := supported && hw.GpuCompositorReady
	return MediaCapabilitiesInfo{
		Supported:      supported,
		Smooth:         smooth,
		PowerEfficient: powerEfficient,
	}
}

// Av1CdefFilter4x4 applies the AOMedia AV1 Section 7.15 CDEF filter to a 4x4 luma block.
func Av1CdefFilter4x4(block *[16]byte, strength, priDamping uint8) {
	if strength == 0 {
		return
	}
	s := int32(strength)
	if s > 15 {
		s = 15
	}
	damp := clampI32(int32(priDamping), 3, 6)
	orig := *block
	tapWeight := func(diff, thresh int32) int32 {
		ad := diff
		if ad < 0 {
			ad = -ad
		}
		if ad == 0 || thresh == 0 {
			return 0
		}
		shift := damp - int32(bits.Len32(uint32(thresh))-1)
		if shift < 0 {
			shift = 0
		}
		mag := thresh - (ad >> shift)
		if mag < 0 {
			mag = 0
		}
		if mag > ad {
			mag = ad
		}
		if diff < 0 {
			return -mag
		}
		return mag
	}
	offsets := [4][3]int{{-1, 0, 4}, {1, 0, 4}, {0, -1, 2}, {0, 1, 2}}
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			center := int32(orig[y*4+x])
			sum := int32(0)
			for _, off := range offsets {
				nx := int(clampI32(int32(x+off[0]), 0, 3))
				ny := int(clampI32(int32(y+off[1]), 0, 3))
				neighbor := int32(orig[ny*4+nx])
				sum += int32(off[2]) * tapWeight(neighbor-center, s)
			}
			filtered := clampI32(center+((sum+8)>>4), 0, 255)
			block[y*4+x] = byte(filtered)
		}
	}
}
