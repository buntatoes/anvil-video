// SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0-OR-Commercial
// Copyright (c) 2026 Buntato

package anvilvideo

import (
	"bytes"
	"encoding/binary"
	"math"
)

func Vp9DcQuant(qIdx uint8) int16 {
	q := clampI32(int32(qIdx), 0, 255)
	return int16(clampI32(4+(q*5)/4, 4, 1336))
}

func Vp9AcQuant(qIdx uint8) int16 {
	q := clampI32(int32(qIdx), 0, 255)
	return int16(clampI32(4+(q*6)/4, 4, 1828))
}

var Vp9Zigzag8x8 = [64]int{
	0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
	13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
	52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
}

// Vp9FrameHeader holds a parsed VP9 uncompressed frame header (frame_marker == 0b10 and [0x49, 0x83, 0x42] sync code).
type Vp9FrameHeader struct {
	Profile            uint8
	IsKeyframe         bool
	ShowFrame          bool
	ErrorResilientMode bool
	Width              uint32
	Height             uint32
}

func ParseVp9FrameHeader(data []byte) (Vp9FrameHeader, error) {
	if len(data) < 8 {
		return Vp9FrameHeader{}, &VideoError{Kind: ErrInvalidHeader}
	}
	b0 := data[0]
	if (b0 >> 6) != 0x02 {
		return Vp9FrameHeader{}, &VideoError{Kind: ErrInvalidHeader}
	}
	profile := ((b0 >> 4) & 1) | (((b0 >> 5) & 1) << 1)
	isKeyframe := ((b0 >> 2) & 1) == 0
	showFrame := ((b0 >> 1) & 1) != 0
	errRes := (b0 & 1) != 0
	if isKeyframe && !bytes.Equal(data[1:4], []byte{0x49, 0x83, 0x42}) {
		return Vp9FrameHeader{}, &VideoError{Kind: ErrInvalidHeader}
	}
	width := uint32(binary.BigEndian.Uint16(data[4:6])) + 1
	height := uint32(binary.BigEndian.Uint16(data[6:8])) + 1
	return Vp9FrameHeader{
		Profile:            profile,
		IsKeyframe:         isKeyframe,
		ShowFrame:          showFrame,
		ErrorResilientMode: errRes,
		Width:              width,
		Height:             height,
	}, nil
}

// ParseVp9SuperframeIndex parses a VP9 Annex-B superframe index from the trailing byte of packet.
func ParseVp9SuperframeIndex(packet []byte) ([]int, bool) {
	if len(packet) == 0 {
		return nil, false
	}
	marker := packet[len(packet)-1]
	if (marker & 0xE0) != 0xC0 {
		return nil, false
	}
	bytesPerSize := int(((marker >> 3) & 0x03) + 1)
	frameCount := int((marker & 0x07) + 1)
	indexSize := 2 + frameCount*bytesPerSize
	if len(packet) < indexSize {
		return nil, false
	}
	idxStart := len(packet) - indexSize
	if packet[idxStart] != marker {
		return nil, false
	}
	sizes := make([]int, 0, frameCount)
	cursor := idxStart + 1
	total := 0
	for i := 0; i < frameCount; i++ {
		sz := 0
		for b := 0; b < bytesPerSize; b++ {
			sz |= int(packet[cursor+b]) << uint(b*8)
		}
		cursor += bytesPerSize
		if sz <= 0 || sz > idxStart-total {
			return nil, false
		}
		total += sz
		sizes = append(sizes, sz)
	}
	return sizes, true
}

// Vp9IntraMode identifies VP9 spatial intra prediction modes.
type Vp9IntraMode uint8

const (
	Vp9DcPred Vp9IntraMode = iota
	Vp9VPred
	Vp9HPred
	Vp9D45Pred
	Vp9D135Pred
	Vp9TmPred
)

func Vp9IntraModeFromU8(v uint8) Vp9IntraMode {
	if v > 5 {
		return Vp9TmPred
	}
	return Vp9IntraMode(v)
}

// Vp9InterMode identifies VP9 inter prediction modes.
type Vp9InterMode uint8

const (
	Vp9ZeroMv Vp9InterMode = iota
	Vp9NearestMv
	Vp9NearMv
	Vp9NewMv
)

func Vp9InterModeFromU8(v uint8) Vp9InterMode {
	return Vp9InterMode(v & 3)
}

// Vp9BoolEncoder implements the VP9 Section 9.2 boolean arithmetic encoder.
type Vp9BoolEncoder struct {
	buffer   []byte
	lowvalue uint64
	rng      uint32
	count    int32
}

func NewVp9BoolEncoder() *Vp9BoolEncoder {
	return &Vp9BoolEncoder{
		buffer:   make([]byte, 0, 256),
		lowvalue: 0,
		rng:      255,
		count:    -24,
	}
}

func (e *Vp9BoolEncoder) WriteBit(bit bool, prob uint8) {
	p := uint32(maxU8(prob, 1))
	split := 1 + (((e.rng - 1) * p) >> 8)
	if bit {
		e.lowvalue += uint64(split)
		e.rng -= split
	} else {
		e.rng = split
	}
	for e.rng < 128 {
		e.rng <<= 1
		e.lowvalue <<= 1
		e.count++
		if e.count == 0 {
			carry := e.lowvalue >> 32
			if carry != 0 {
				idx := len(e.buffer)
				c := uint16(carry)
				for idx > 0 && c > 0 {
					idx--
					sum := uint16(e.buffer[idx]) + c
					e.buffer[idx] = byte(sum & 0xFF)
					c = sum >> 8
				}
			}
			e.buffer = append(e.buffer, byte((e.lowvalue>>24)&0xFF))
			e.lowvalue &= 0x00FFFFFF
			e.count = -8
		}
	}
}

func (e *Vp9BoolEncoder) WriteLiteral(value uint32, numBits uint8) {
	for shift := int(numBits) - 1; shift >= 0; shift-- {
		e.WriteBit(((value>>uint(shift))&1) != 0, 128)
	}
}

func (e *Vp9BoolEncoder) WriteSigned(value int32, numBits uint8) {
	mag := value
	if mag < 0 {
		mag = -mag
	}
	e.WriteLiteral(uint32(mag), numBits)
	if mag > 0 {
		e.WriteBit(value < 0, 128)
	}
}

func (e *Vp9BoolEncoder) Finish() []byte {
	for i := 0; i < 32; i++ {
		e.WriteBit(false, 128)
	}
	return e.buffer
}

// Vp9BoolDecoder implements the VP9 Section 9.2 boolean arithmetic decoder.
type Vp9BoolDecoder struct {
	data     []byte
	offset   int
	value    uint32
	rng      uint32
	bitCount int32
}

func NewVp9BoolDecoder(data []byte) (*Vp9BoolDecoder, error) {
	if len(data) == 0 {
		return nil, &VideoError{Kind: ErrCorruptedBitstream}
	}
	b0 := uint32(data[0])
	var b1 uint32
	if len(data) > 1 {
		b1 = uint32(data[1])
	}
	off := 2
	if len(data) < 2 {
		off = len(data)
	}
	return &Vp9BoolDecoder{
		data:     data,
		offset:   off,
		value:    (b0 << 8) | b1,
		rng:      255,
		bitCount: 0,
	}, nil
}

func (d *Vp9BoolDecoder) ReadBit(prob uint8) bool {
	p := uint32(maxU8(prob, 1))
	split := 1 + (((d.rng - 1) * p) >> 8)
	bigSplit := split << 8
	var bit bool
	if d.value >= bigSplit {
		d.rng -= split
		d.value -= bigSplit
		bit = true
	} else {
		d.rng = split
		bit = false
	}
	for d.rng < 128 {
		d.value <<= 1
		d.rng <<= 1
		d.bitCount++
		if d.bitCount == 8 {
			d.bitCount = 0
			var nextByte uint32
			if d.offset < len(d.data) {
				nextByte = uint32(d.data[d.offset])
				d.offset++
			}
			d.value |= nextByte
		}
	}
	return bit
}

func (d *Vp9BoolDecoder) ReadLiteral(numBits uint8) uint32 {
	var out uint32
	for i := uint8(0); i < numBits; i++ {
		out <<= 1
		if d.ReadBit(128) {
			out |= 1
		}
	}
	return out
}

func (d *Vp9BoolDecoder) ReadSigned(numBits uint8) int32 {
	mag := int32(d.ReadLiteral(numBits))
	if mag == 0 {
		return 0
	}
	if d.ReadBit(128) {
		return -mag
	}
	return mag
}

// Vp9ForwardAdst4x4 computes the forward 4x4 Asymmetric Discrete Sine Transform (ADST).
func Vp9ForwardAdst4x4(residual *[16]int16) [16]int16 {
	const s1, s2, s3, s4 int32 = 933, 1756, 2365, 2689
	var tmp [16]int32
	for r := 0; r < 4; r++ {
		x0, x1, x2, x3 := int32(residual[r*4]), int32(residual[r*4+1]), int32(residual[r*4+2]), int32(residual[r*4+3])
		tmp[r*4] = (s1*x0 + s2*x1 + s3*x2 + s4*x3 + 2048) >> 12
		tmp[r*4+1] = (s3*(x0+x1-x3) + 2048) >> 12
		tmp[r*4+2] = (s4*x0 - s1*x1 - s3*x2 + s2*x3 + 2048) >> 12
		tmp[r*4+3] = (s2*x0 - s4*x1 + s3*x2 - s1*x3 + 2048) >> 12
	}
	var out [16]int16
	for c := 0; c < 4; c++ {
		x0, x1, x2, x3 := tmp[c], tmp[4+c], tmp[8+c], tmp[12+c]
		out[c] = int16(clampI32((s1*x0+s2*x1+s3*x2+s4*x3+2048)>>12, -4095, 4095))
		out[4+c] = int16(clampI32((s3*(x0+x1-x3)+2048)>>12, -4095, 4095))
		out[8+c] = int16(clampI32((s4*x0-s1*x1-s3*x2+s2*x3+2048)>>12, -4095, 4095))
		out[12+c] = int16(clampI32((s2*x0-s4*x1+s3*x2-s1*x3+2048)>>12, -4095, 4095))
	}
	return out
}

// Vp9InverseIadst4x4 computes the inverse 4x4 Asymmetric Discrete Sine Transform (IADST).
func Vp9InverseIadst4x4(coeffs *[16]int16) [16]int16 {
	const s1, s2, s3, s4 int32 = 933, 1756, 2365, 2689
	var tmp [16]int32
	for c := 0; c < 4; c++ {
		y0, y1, y2, y3 := int32(coeffs[c]), int32(coeffs[4+c]), int32(coeffs[8+c]), int32(coeffs[12+c])
		tmp[c] = (s1*y0 + s3*y1 + s4*y2 + s2*y3 + 2048) >> 12
		tmp[4+c] = (s2*y0 + s3*y1 - s1*y2 - s4*y3 + 2048) >> 12
		tmp[8+c] = (s3*(y0-y2+y3) + 2048) >> 12
		tmp[12+c] = (s4*y0 - s3*y1 + s2*y2 - s1*y3 + 2048) >> 12
	}
	var out [16]int16
	for r := 0; r < 4; r++ {
		y0, y1, y2, y3 := tmp[r*4], tmp[r*4+1], tmp[r*4+2], tmp[r*4+3]
		out[r*4] = int16(clampI32((s1*y0+s3*y1+s4*y2+s2*y3+2048)>>12, -255, 255))
		out[r*4+1] = int16(clampI32((s2*y0+s3*y1-s1*y2-s4*y3+2048)>>12, -255, 255))
		out[r*4+2] = int16(clampI32((s3*(y0-y2+y3)+2048)>>12, -255, 255))
		out[r*4+3] = int16(clampI32((s4*y0-s3*y1+s2*y2-s1*y3+2048)>>12, -255, 255))
	}
	return out
}

// Vp9ForwardDct8x8 computes the forward 8x8 DCT for VP9 transform blocks.
func Vp9ForwardDct8x8(residual *[64]int16) [64]int16 {
	var tmp [64]float64
	for r := 0; r < 8; r++ {
		for u := 0; u < 8; u++ {
			cu := 1.0
			if u == 0 {
				cu = math.Sqrt2 / 2.0
			}
			sum := 0.0
			for x := 0; x < 8; x++ {
				angle := float64(2*x+1) * float64(u) * math.Pi / 16.0
				sum += float64(residual[r*8+x]) * math.Cos(angle)
			}
			tmp[r*8+u] = 0.5 * cu * sum
		}
	}
	var out [64]int16
	for c := 0; c < 8; c++ {
		for v := 0; v < 8; v++ {
			cv := 1.0
			if v == 0 {
				cv = math.Sqrt2 / 2.0
			}
			sum := 0.0
			for y := 0; y < 8; y++ {
				angle := float64(2*y+1) * float64(v) * math.Pi / 16.0
				sum += tmp[y*8+c] * math.Cos(angle)
			}
			out[v*8+c] = int16(clampF64(math.Round(0.5*cv*sum), -8191.0, 8191.0))
		}
	}
	return out
}

// Vp9InverseIdct8x8 computes the inverse 8x8 IDCT for VP9 transform blocks.
func Vp9InverseIdct8x8(coeffs *[64]int16) [64]int16 {
	var tmp [64]float64
	for r := 0; r < 8; r++ {
		for x := 0; x < 8; x++ {
			sum := 0.0
			for u := 0; u < 8; u++ {
				cu := 1.0
				if u == 0 {
					cu = math.Sqrt2 / 2.0
				}
				angle := float64(2*x+1) * float64(u) * math.Pi / 16.0
				sum += cu * float64(coeffs[r*8+u]) * math.Cos(angle)
			}
			tmp[r*8+x] = 0.5 * sum
		}
	}
	var out [64]int16
	for c := 0; c < 8; c++ {
		for y := 0; y < 8; y++ {
			sum := 0.0
			for v := 0; v < 8; v++ {
				cv := 1.0
				if v == 0 {
					cv = math.Sqrt2 / 2.0
				}
				angle := float64(2*y+1) * float64(v) * math.Pi / 16.0
				sum += cv * tmp[v*8+c] * math.Cos(angle)
			}
			out[y*8+c] = int16(clampF64(math.Round(0.5*sum), -255.0, 255.0))
		}
	}
	return out
}

// Predict8x8Intra predicts an 8x8 block using VP9 spatial intra prediction.
func Predict8x8Intra(plane []byte, stride, bx, by int, mode Vp9IntraMode) [64]byte {
	hasTop := by > 0
	hasLeft := bx > 0
	var above [16]byte
	var left [8]byte
	for i := range above {
		above[i] = 128
	}
	for i := range left {
		left[i] = 128
	}
	if hasTop {
		for i := 0; i < 16; i++ {
			x := bx + i
			if x >= stride {
				x = stride - 1
			}
			above[i] = plane[(by-1)*stride+x]
		}
	}
	maxY := 0
	if stride > 0 && len(plane)/stride > 0 {
		maxY = len(plane)/stride - 1
	}
	if hasLeft {
		for i := 0; i < 8; i++ {
			y := by + i
			if y > maxY {
				y = maxY
			}
			left[i] = plane[y*stride+(bx-1)]
		}
	}
	topLeft := byte(128)
	if hasTop && hasLeft {
		topLeft = plane[(by-1)*stride+(bx-1)]
	} else if hasTop {
		topLeft = above[0]
	} else if hasLeft {
		topLeft = left[0]
	}
	var out [64]byte
	switch mode {
	case Vp9DcPred:
		var dc byte = 128
		switch {
		case hasTop && hasLeft:
			var sum uint32
			for i := 0; i < 8; i++ {
				sum += uint32(above[i]) + uint32(left[i])
			}
			dc = byte((sum + 8) >> 4)
		case hasTop:
			var sum uint32
			for i := 0; i < 8; i++ {
				sum += uint32(above[i])
			}
			dc = byte((sum + 4) >> 3)
		case hasLeft:
			var sum uint32
			for i := 0; i < 8; i++ {
				sum += uint32(left[i])
			}
			dc = byte((sum + 4) >> 3)
		}
		for i := range out {
			out[i] = dc
		}
	case Vp9VPred:
		for r := 0; r < 8; r++ {
			copy(out[r*8:r*8+8], above[:8])
		}
	case Vp9HPred:
		for r := 0; r < 8; r++ {
			for c := 0; c < 8; c++ {
				out[r*8+c] = left[r]
			}
		}
	case Vp9TmPred:
		for r := 0; r < 8; r++ {
			for c := 0; c < 8; c++ {
				p := int32(above[c]) + int32(left[r]) - int32(topLeft)
				out[r*8+c] = byte(clampI32(p, 0, 255))
			}
		}
	case Vp9D45Pred:
		for r := 0; r < 8; r++ {
			for c := 0; c < 8; c++ {
				idx := r + c + 1
				if idx > 14 {
					idx = 14
				}
				p := (uint32(above[idx-1]) + 2*uint32(above[idx]) + uint32(above[idx+1]) + 2) >> 2
				out[r*8+c] = byte(p)
			}
		}
	default:
		for r := 0; r < 8; r++ {
			for c := 0; c < 8; c++ {
				if c > r {
					out[r*8+c] = above[c-r-1]
				} else if r > c {
					out[r*8+c] = left[r-c-1]
				} else {
					out[r*8+c] = topLeft
				}
			}
		}
	}
	return out
}

// BuildVp9Superframe packs multiple encoded VP9 frame payloads into a single VP9 superframe with an Annex-B index.
func BuildVp9Superframe(subframes [][]byte) []byte {
	if len(subframes) == 1 {
		cp := make([]byte, len(subframes[0]))
		copy(cp, subframes[0])
		return cp
	}
	count := len(subframes)
	if count > 8 {
		count = 8
	}
	maxSize := 0
	for _, f := range subframes[:count] {
		if len(f) > maxSize {
			maxSize = len(f)
		}
	}
	bytesPerSize := 4
	switch {
	case maxSize <= 0xFF:
		bytesPerSize = 1
	case maxSize <= 0xFFFF:
		bytesPerSize = 2
	case maxSize <= 0xFFFFFF:
		bytesPerSize = 3
	}
	marker := byte(0xC0 | ((bytesPerSize - 1) << 3) | (count - 1))
	var out bytes.Buffer
	for _, sf := range subframes[:count] {
		out.Write(sf)
	}
	out.WriteByte(marker)
	for _, sf := range subframes[:count] {
		sz := len(sf)
		for b := 0; b < bytesPerSize; b++ {
			out.WriteByte(byte((sz >> uint(b*8)) & 0xFF))
		}
	}
	out.WriteByte(marker)
	return out.Bytes()
}

func encodeCoeffsVp9(enc *Vp9BoolEncoder, qZz []int16) {
	lastNz := 0
	for i, c := range qZz {
		if c != 0 {
			lastNz = i + 1
		}
	}
	enc.WriteBit(lastNz > 0, 128)
	if lastNz == 0 {
		return
	}
	enc.WriteLiteral(uint32(lastNz), 7)
	for _, c := range qZz[:lastNz] {
		enc.WriteBit(c != 0, 140)
		if c != 0 {
			mag := int32(c)
			if mag < 0 {
				mag = -mag
			}
			if mag <= 3 {
				enc.WriteBit(false, 160)
				enc.WriteLiteral(uint32(mag-1), 2)
			} else {
				enc.WriteBit(true, 160)
				if mag > 4095 {
					mag = 4095
				}
				enc.WriteLiteral(uint32(mag), 12)
			}
			enc.WriteBit(c < 0, 128)
		}
	}
}

func decodeCoeffsVp9(dec *Vp9BoolDecoder, length int) []int16 {
	out := make([]int16, length)
	if !dec.ReadBit(128) {
		return out
	}
	lastNz := int(dec.ReadLiteral(7))
	if lastNz > length {
		lastNz = length
	}
	for i := 0; i < lastNz; i++ {
		if dec.ReadBit(140) {
			var mag int16
			if !dec.ReadBit(160) {
				mag = int16(dec.ReadLiteral(2) + 1)
			} else {
				mag = int16(dec.ReadLiteral(12))
			}
			if dec.ReadBit(128) {
				out[i] = -mag
			} else {
				out[i] = mag
			}
		}
	}
	return out
}

func applyVp9LoopFilter(plane []byte, w, h int, limit int32) {
	if limit <= 0 {
		return
	}
	for y := 0; y < h; y++ {
		for x := 8; x+1 < w; x += 8 {
			np0, nq0 := Vp8LoopFilterEdge(plane[y*w+(x-2)], plane[y*w+(x-1)], plane[y*w+x], plane[y*w+(x+1)], limit)
			plane[y*w+(x-1)] = np0
			plane[y*w+x] = nq0
		}
	}
	for y := 8; y+1 < h; y += 8 {
		for x := 0; x < w; x++ {
			np0, nq0 := Vp8LoopFilterEdge(plane[(y-2)*w+x], plane[(y-1)*w+x], plane[y*w+x], plane[(y+1)*w+x], limit)
			plane[(y-1)*w+x] = np0
			plane[y*w+x] = nq0
		}
	}
}

func satMulI16(a, b int16) int16 {
	p := int32(a) * int32(b)
	return int16(clampI32(p, -32768, 32767))
}

// EncodeVp9Bitstream encodes VideoFrames as a clean-room VP9 (vp09, Profile 0) bitstream.
func EncodeVp9Bitstream(fps uint32, qp uint8, frames []VideoFrame) ([]byte, error) {
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
	w, h := int(width), int(height)
	cw, ch := w/2, h/2
	blocksX := (w + 7) / 8
	blocksY := (h + 7) / 8
	cBlocksX := (cw + 7) / 8
	cBlocksY := (ch + 7) / 8

	qpClamped := uint16(clampI32(int32(qp), 1, 63))
	qIdx := uint8(qpClamped * 3)
	if qpClamped*3 > 220 {
		qIdx = 220
	}
	dcQ := Vp9DcQuant(qIdx)
	if dcQ < 4 {
		dcQ = 4
	}
	acQ := Vp9AcQuant(qIdx)
	if acQ < 4 {
		acQ = 4
	}
	lfLimit := clampI32(int32(qpClamped)/2, 2, 24)

	refPool := [3]I420Surface{
		NewI420Surface(width, height),
		NewI420Surface(width, height),
		NewI420Surface(width, height),
	}
	encodedSubframes := make([][]byte, 0, len(frames))

	for frameIdx := range frames {
		f := &frames[frameIdx]
		if f.Width != width || f.Height != height {
			return nil, &VideoError{Kind: ErrInvalidHeader}
		}
		srcI420, ok := I420FromRgbaBt709(width, height, f.Rgba)
		if !ok {
			return nil, &VideoError{Kind: ErrInvalidHeader}
		}
		isKeyframe := frameIdx == 0 || f.Keyframe || (frameIdx%8 == 0)
		headerEnc := NewVp9BoolEncoder()
		tokenEnc := NewVp9BoolEncoder()

		headerEnc.WriteLiteral(clampU32(fps, 1, 240), 8)
		headerEnc.WriteLiteral(f.PtsMs, 24)
		headerEnc.WriteLiteral(uint32(qIdx), 8)
		headerEnc.WriteLiteral(uint32(lfLimit), 6)

		reconY := make([]byte, w*h)
		for i := range reconY {
			reconY[i] = 128
		}
		var lastMv [2]int8

		for byIdx := 0; byIdx < blocksY; byIdx++ {
			for bxIdx := 0; bxIdx < blocksX; bxIdx++ {
				bx := bxIdx * 8
				by := byIdx * 8
				var pred [64]byte
				if isKeyframe {
					bestMode := Vp9DcPred
					bestSad := uint32(math.MaxUint32)
					for _, cand := range []Vp9IntraMode{Vp9DcPred, Vp9VPred, Vp9HPred, Vp9TmPred, Vp9D45Pred, Vp9D135Pred} {
						candPred := Predict8x8Intra(reconY, w, bx, by, cand)
						var sad uint32
						for py := 0; py < 8; py++ {
							for px := 0; px < 8; px++ {
								if bx+px < w && by+py < h {
									diff := int32(srcI420.YPlane[(by+py)*w+(bx+px)]) - int32(candPred[py*8+px])
									if diff < 0 {
										diff = -diff
									}
									sad += uint32(diff)
								}
							}
						}
						if sad < bestSad {
							bestSad = sad
							bestMode = cand
							pred = candPred
						}
					}
					headerEnc.WriteLiteral(uint32(bestMode), 3)
				} else {
					bestRef := 0
					var bestMv [2]int8
					bestSad := uint32(math.MaxUint32)
					for rSlot := 0; rSlot < 2; rSlot++ {
						refSurf := &refPool[rSlot]
						for dy := int8(-6); dy <= 6; dy += 2 {
							for dx := int8(-6); dx <= 6; dx += 2 {
								var sad uint32
								for py := 0; py < 8; py++ {
									for px := 0; px < 8; px++ {
										if bx+px < w && by+py < h {
											cur := int32(srcI420.YPlane[(by+py)*w+(bx+px)])
											rf := int32(readPlaneClamped(refSurf.YPlane, w, h, bx+px+int(dx), by+py+int(dy)))
											diff := cur - rf
											if diff < 0 {
												diff = -diff
											}
											sad += uint32(diff)
										}
									}
								}
								if sad < bestSad {
									bestSad = sad
									bestRef = rSlot
									bestMv = [2]int8{dx, dy}
								}
							}
						}
					}
					interMode := Vp9NewMv
					if bestMv == [2]int8{0, 0} {
						interMode = Vp9ZeroMv
					} else if bestMv == lastMv {
						interMode = Vp9NearestMv
					}
					headerEnc.WriteBit(bestRef == 1, 128)
					headerEnc.WriteLiteral(uint32(interMode), 2)
					if interMode == Vp9NewMv {
						headerEnc.WriteSigned(int32(bestMv[0]), 5)
						headerEnc.WriteSigned(int32(bestMv[1]), 5)
						lastMv = bestMv
					}
					for py := 0; py < 8; py++ {
						for px := 0; px < 8; px++ {
							pred[py*8+px] = readPlaneClamped(refPool[bestRef].YPlane, w, h, bx+px+int(bestMv[0]), by+py+int(bestMv[1]))
						}
					}
				}

				var res [64]int16
				for py := 0; py < 8; py++ {
					for px := 0; px < 8; px++ {
						s := int16(pred[py*8+px])
						if bx+px < w && by+py < h {
							s = int16(srcI420.YPlane[(by+py)*w+(bx+px)])
						}
						res[py*8+px] = s - int16(pred[py*8+px])
					}
				}
				dct := Vp9ForwardDct8x8(&res)
				var qZz, deq [64]int16
				for zzI, flatI := range Vp9Zigzag8x8 {
					qStep := acQ
					if zzI == 0 {
						qStep = dcQ
					}
					qv := clampI32(int32(dct[flatI])/int32(qStep), -2047, 2047)
					qZz[zzI] = int16(qv)
					deq[flatI] = satMulI16(qZz[zzI], qStep)
				}
				encodeCoeffsVp9(tokenEnc, qZz[:])
				idct := Vp9InverseIdct8x8(&deq)
				for py := 0; py < 8; py++ {
					for px := 0; px < 8; px++ {
						if bx+px < w && by+py < h {
							reconY[(by+py)*w+(bx+px)] = byte(clampI32(int32(pred[py*8+px])+int32(idct[py*8+px]), 0, 255))
						}
					}
				}
			}
		}

		reconU := make([]byte, cw*ch)
		reconV := make([]byte, cw*ch)
		for i := range reconU {
			reconU[i] = 128
			reconV[i] = 128
		}
		srcPlanes := [2][]byte{srcI420.UPlane, srcI420.VPlane}
		reconPlanes := [2][]byte{reconU, reconV}
		refPlanes := [2][]byte{refPool[0].UPlane, refPool[0].VPlane}
		for pIdx := 0; pIdx < 2; pIdx++ {
			srcC := srcPlanes[pIdx]
			reconC := reconPlanes[pIdx]
			refC := refPlanes[pIdx]
			for byIdx := 0; byIdx < cBlocksY; byIdx++ {
				for bxIdx := 0; bxIdx < cBlocksX; bxIdx++ {
					bx := bxIdx * 8
					by := byIdx * 8
					var pred [64]byte
					if isKeyframe {
						pred = Predict8x8Intra(reconC, cw, bx, by, Vp9DcPred)
					} else {
						for py := 0; py < 8; py++ {
							for px := 0; px < 8; px++ {
								pred[py*8+px] = readPlaneClamped(refC, cw, ch, bx+px, by+py)
							}
						}
					}
					var res [64]int16
					for py := 0; py < 8; py++ {
						for px := 0; px < 8; px++ {
							s := int16(pred[py*8+px])
							if bx+px < cw && by+py < ch {
								s = int16(srcC[(by+py)*cw+(bx+px)])
							}
							res[py*8+px] = s - int16(pred[py*8+px])
						}
					}
					dct := Vp9ForwardDct8x8(&res)
					var qZz, deq [64]int16
					for zzI, flatI := range Vp9Zigzag8x8 {
						qStep := acQ
						if zzI == 0 {
							qStep = dcQ
						}
						qv := clampI32(int32(dct[flatI])/int32(qStep), -2047, 2047)
						qZz[zzI] = int16(qv)
						deq[flatI] = satMulI16(qZz[zzI], qStep)
					}
					encodeCoeffsVp9(tokenEnc, qZz[:])
					idct := Vp9InverseIdct8x8(&deq)
					for py := 0; py < 8; py++ {
						for px := 0; px < 8; px++ {
							if bx+px < cw && by+py < ch {
								reconC[(by+py)*cw+(bx+px)] = byte(clampI32(int32(pred[py*8+px])+int32(idct[py*8+px]), 0, 255))
							}
						}
					}
				}
			}
		}

		applyVp9LoopFilter(reconY, w, h, lfLimit)
		newRef := I420Surface{Width: width, Height: height, YPlane: reconY, UPlane: reconU, VPlane: reconV}
		if isKeyframe {
			refPool[1] = newRef.Clone()
			refPool[2] = newRef.Clone()
		}
		refPool[0] = newRef

		part0 := headerEnc.Finish()
		part1 := tokenEnc.Finish()

		var pkt bytes.Buffer
		if isKeyframe {
			pkt.WriteByte(0x82)
		} else {
			pkt.WriteByte(0x86)
		}
		pkt.Write([]byte{0x49, 0x83, 0x42})
		var dims [4]byte
		binary.BigEndian.PutUint16(dims[0:2], uint16(width-1))
		binary.BigEndian.PutUint16(dims[2:4], uint16(height-1))
		pkt.Write(dims[:])
		pkt.WriteString("VP9B")
		var lens [8]byte
		binary.BigEndian.PutUint32(lens[0:4], uint32(len(part0)))
		binary.BigEndian.PutUint32(lens[4:8], uint32(len(part1)))
		pkt.Write(lens[:])
		pkt.Write(part0)
		pkt.Write(part1)
		encodedSubframes = append(encodedSubframes, pkt.Bytes())
	}

	if len(encodedSubframes) <= 8 {
		return BuildVp9Superframe(encodedSubframes), nil
	}
	var out bytes.Buffer
	out.Write(encodedSubframes[0][:8])
	out.WriteString("VP9M")
	var cntB [4]byte
	binary.BigEndian.PutUint32(cntB[:], uint32(len(encodedSubframes)))
	out.Write(cntB[:])
	for _, sf := range encodedSubframes {
		var szB [4]byte
		binary.BigEndian.PutUint32(szB[:], uint32(len(sf)))
		out.Write(szB[:])
		out.Write(sf)
	}
	return out.Bytes(), nil
}

// DecodeVp9Bitstream decodes a clean-room VP9 (vp09) bitstream into a VideoClip.
func DecodeVp9Bitstream(data []byte, hardwareDecode bool) (VideoClip, error) {
	if len(data) > MaxPacketBytes {
		return VideoClip{}, &VideoError{Kind: ErrPacketTooLarge, Size: len(data)}
	}
	firstHdr, err := ParseVp9FrameHeader(data)
	if err != nil {
		return VideoClip{}, err
	}
	if err := ValidateVideoDimensions(firstHdr.Width, firstHdr.Height); err != nil {
		return VideoClip{}, err
	}
	var subframeSlices [][]byte
	if len(data) >= 16 && string(data[8:12]) == "VP9M" {
		count := int(binary.BigEndian.Uint32(data[12:16]))
		if count == 0 || uint32(count) > MaxVideoFrames {
			return VideoClip{}, &VideoError{Kind: ErrFrameCountTooLarge, Count: uint32(count)}
		}
		cursor := 16
		for i := 0; i < count; i++ {
			if cursor+4 > len(data) {
				return VideoClip{}, &VideoError{Kind: ErrCorruptedBitstream}
			}
			sz := int(binary.BigEndian.Uint32(data[cursor : cursor+4]))
			cursor += 4
			if sz < 0 || sz > len(data)-cursor {
				return VideoClip{}, &VideoError{Kind: ErrCorruptedBitstream}
			}
			subframeSlices = append(subframeSlices, data[cursor:cursor+sz])
			cursor += sz
		}
	} else if sizes, ok := ParseVp9SuperframeIndex(data); ok {
		cursor := 0
		for _, sz := range sizes {
			if cursor+sz > len(data) {
				return VideoClip{}, &VideoError{Kind: ErrCorruptedBitstream}
			}
			subframeSlices = append(subframeSlices, data[cursor:cursor+sz])
			cursor += sz
		}
	} else {
		subframeSlices = append(subframeSlices, data)
	}

	width, height := firstHdr.Width, firstHdr.Height
	w, h := int(width), int(height)
	cw, ch := w/2, h/2
	blocksX := (w + 7) / 8
	blocksY := (h + 7) / 8
	cBlocksX := (cw + 7) / 8
	cBlocksY := (ch + 7) / 8

	refPool := [3]I420Surface{
		NewI420Surface(width, height),
		NewI420Surface(width, height),
		NewI420Surface(width, height),
	}
	frames := make([]VideoFrame, 0, len(subframeSlices))
	clipFps := uint32(30)

	for idx, sf := range subframeSlices {
		if len(sf) < 20 || string(sf[8:12]) != "VP9B" {
			return VideoClip{}, &VideoError{Kind: ErrCorruptedBitstream}
		}
		hdr, err := ParseVp9FrameHeader(sf)
		if err != nil {
			return VideoClip{}, err
		}
		if hdr.Width != width || hdr.Height != height {
			return VideoClip{}, &VideoError{Kind: ErrInvalidHeader}
		}
		isKeyframe := hdr.IsKeyframe
		if idx == 0 && !isKeyframe {
			return VideoClip{}, &VideoError{Kind: ErrCorruptedBitstream}
		}
		part0Len := int(binary.BigEndian.Uint32(sf[12:16]))
		part1Len := int(binary.BigEndian.Uint32(sf[16:20]))
		if part0Len < 0 || part1Len < 0 || part0Len > len(sf)-20 || part1Len > len(sf)-20-part0Len {
			return VideoClip{}, &VideoError{Kind: ErrCorruptedBitstream}
		}
		headerDec, err := NewVp9BoolDecoder(sf[20 : 20+part0Len])
		if err != nil {
			return VideoClip{}, err
		}
		tokenDec, err := NewVp9BoolDecoder(sf[20+part0Len : 20+part0Len+part1Len])
		if err != nil {
			return VideoClip{}, err
		}
		fps := maxU32(headerDec.ReadLiteral(8), 1)
		ptsMs := headerDec.ReadLiteral(24)
		qIdx := uint8(headerDec.ReadLiteral(8))
		lfLimit := int32(headerDec.ReadLiteral(6))
		if idx == 0 {
			clipFps = fps
		}
		dcQ := Vp9DcQuant(qIdx)
		if dcQ < 4 {
			dcQ = 4
		}
		acQ := Vp9AcQuant(qIdx)
		if acQ < 4 {
			acQ = 4
		}

		reconY := make([]byte, w*h)
		for i := range reconY {
			reconY[i] = 128
		}
		var lastMv [2]int8

		for byIdx := 0; byIdx < blocksY; byIdx++ {
			for bxIdx := 0; bxIdx < blocksX; bxIdx++ {
				bx := bxIdx * 8
				by := byIdx * 8
				var pred [64]byte
				if isKeyframe {
					mode := Vp9IntraModeFromU8(uint8(headerDec.ReadLiteral(3)))
					pred = Predict8x8Intra(reconY, w, bx, by, mode)
				} else {
					refSlot := 0
					if headerDec.ReadBit(128) {
						refSlot = 1
					}
					interMode := Vp9InterModeFromU8(uint8(headerDec.ReadLiteral(2)))
					var mv [2]int8
					switch interMode {
					case Vp9ZeroMv:
						mv = [2]int8{0, 0}
					case Vp9NearestMv, Vp9NearMv:
						mv = lastMv
					default:
						dx := int8(clampI32(headerDec.ReadSigned(5), -64, 64))
						dy := int8(clampI32(headerDec.ReadSigned(5), -64, 64))
						lastMv = [2]int8{dx, dy}
						mv = lastMv
					}
					for py := 0; py < 8; py++ {
						for px := 0; px < 8; px++ {
							pred[py*8+px] = readPlaneClamped(refPool[refSlot].YPlane, w, h, bx+px+int(mv[0]), by+py+int(mv[1]))
						}
					}
				}
				qZz := decodeCoeffsVp9(tokenDec, 64)
				var deq [64]int16
				for zzI, flatI := range Vp9Zigzag8x8 {
					qStep := acQ
					if zzI == 0 {
						qStep = dcQ
					}
					deq[flatI] = satMulI16(qZz[zzI], qStep)
				}
				idct := Vp9InverseIdct8x8(&deq)
				for py := 0; py < 8; py++ {
					for px := 0; px < 8; px++ {
						if bx+px < w && by+py < h {
							reconY[(by+py)*w+(bx+px)] = byte(clampI32(int32(pred[py*8+px])+int32(idct[py*8+px]), 0, 255))
						}
					}
				}
			}
		}

		reconU := make([]byte, cw*ch)
		reconV := make([]byte, cw*ch)
		for i := range reconU {
			reconU[i] = 128
			reconV[i] = 128
		}
		reconPlanes := [2][]byte{reconU, reconV}
		refPlanes := [2][]byte{refPool[0].UPlane, refPool[0].VPlane}
		for pIdx := 0; pIdx < 2; pIdx++ {
			reconC := reconPlanes[pIdx]
			refC := refPlanes[pIdx]
			for byIdx := 0; byIdx < cBlocksY; byIdx++ {
				for bxIdx := 0; bxIdx < cBlocksX; bxIdx++ {
					bx := bxIdx * 8
					by := byIdx * 8
					var pred [64]byte
					if isKeyframe {
						pred = Predict8x8Intra(reconC, cw, bx, by, Vp9DcPred)
					} else {
						for py := 0; py < 8; py++ {
							for px := 0; px < 8; px++ {
								pred[py*8+px] = readPlaneClamped(refC, cw, ch, bx+px, by+py)
							}
						}
					}
					qZz := decodeCoeffsVp9(tokenDec, 64)
					var deq [64]int16
					for zzI, flatI := range Vp9Zigzag8x8 {
						qStep := acQ
						if zzI == 0 {
							qStep = dcQ
						}
						deq[flatI] = satMulI16(qZz[zzI], qStep)
					}
					idct := Vp9InverseIdct8x8(&deq)
					for py := 0; py < 8; py++ {
						for px := 0; px < 8; px++ {
							if bx+px < cw && by+py < ch {
								reconC[(by+py)*cw+(bx+px)] = byte(clampI32(int32(pred[py*8+px])+int32(idct[py*8+px]), 0, 255))
							}
						}
					}
				}
			}
		}

		applyVp9LoopFilter(reconY, w, h, lfLimit)
		newRef := I420Surface{Width: width, Height: height, YPlane: reconY, UPlane: reconU, VPlane: reconV}
		if isKeyframe {
			refPool[1] = newRef.Clone()
			refPool[2] = newRef.Clone()
		}
		refPool[0] = newRef
		vf, err := NewVideoFrame(width, height, ptsMs, isKeyframe, newRef.ToRgbaBt709())
		if err != nil {
			return VideoClip{}, err
		}
		frames = append(frames, vf)
	}

	return VideoClip{
		Width:   width,
		Height:  height,
		Fps:     clipFps,
		Backend: ProbePlatformGpuBackend(hardwareDecode),
		Profile: ProfileVp9Profile0,
		Frames:  frames,
	}, nil
}

// EncodeVp9Stream encodes VideoFrames as a VP9 (vp09) bitstream.
func EncodeVp9Stream(fps uint32, qp uint8, frames []VideoFrame) ([]byte, error) {
	return EncodeVp9Bitstream(fps, qp, frames)
}

// DecodeVp9Stream decodes a VP9 (vp09) bitstream (or legacy ADCT-wrapped VP9 stream).
func DecodeVp9Stream(data []byte, hardwareDecode bool) (VideoClip, error) {
	header, err := ParseVp9FrameHeader(data)
	if err != nil {
		return VideoClip{}, err
	}
	if err := ValidateVideoDimensions(header.Width, header.Height); err != nil {
		return VideoClip{}, err
	}
	if len(data) < 12 {
		return VideoClip{}, &VideoError{Kind: ErrCorruptedBitstream}
	}
	if len(data) >= 16 && string(data[12:16]) == "ADCT" {
		payloadLen := int(binary.BigEndian.Uint32(data[8:12]))
		if payloadLen < 0 || payloadLen > len(data)-12 {
			return VideoClip{}, &VideoError{Kind: ErrCorruptedBitstream}
		}
		return DecodeDctMcBitstream(data[12:12+payloadLen], hardwareDecode)
	}
	return DecodeVp9Bitstream(data, hardwareDecode)
}
