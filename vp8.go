// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Buntato

package anvilvideo

import (
	"bytes"
	"encoding/binary"
	"math"
)

// Vp8DcQlookup holds the RFC 6386 Section 14.1 DC dequantization lookup table indexed by qi (0..=127).
var Vp8DcQlookup = [128]int16{
	4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16, 17, 17, 18, 19, 20, 20, 21, 21, 22, 22, 23,
	23, 24, 25, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 37, 38, 39, 40, 41, 42, 43, 44,
	45, 46, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
	68, 69, 70, 71, 72, 73, 74, 75, 76, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91,
	93, 95, 96, 98, 100, 101, 102, 104, 106, 108, 110, 112, 114, 116, 118, 122, 124, 126, 128, 130,
	132, 134, 136, 138, 140, 143, 145, 148, 151, 154, 157,
}

// Vp8AcQlookup holds the RFC 6386 Section 14.1 AC dequantization lookup table indexed by qi (0..=127).
var Vp8AcQlookup = [128]int16{
	4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
	29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
	53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
	77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108,
	110, 112, 114, 116, 119, 122, 125, 128, 131, 134, 137, 140, 143, 146, 149, 152, 155, 158, 161,
	164, 167, 170, 173, 177, 181, 185, 189, 193, 197, 201, 205, 209, 213,
}

var Vp8Zigzag = [16]int{0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15}
var Vp8CoeffBands = [16]int{0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7}

var KfYmodeProbs = [4]uint8{145, 156, 163, 128}
var YmodeProbs = [4]uint8{112, 86, 140, 37}
var KfUvModeProbs = [3]uint8{142, 114, 183}
var UvModeProbs = [3]uint8{162, 101, 204}
var DefaultBmodeProbs = [9]uint8{231, 120, 48, 89, 115, 113, 120, 152, 112}

var Pcat1 = [1]uint8{159}
var Pcat2 = [2]uint8{165, 145}
var Pcat3 = [3]uint8{173, 148, 140}
var Pcat4 = [4]uint8{176, 155, 140, 135}
var Pcat5 = [5]uint8{180, 157, 141, 134, 130}
var Pcat6 = [11]uint8{254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129}

var DefaultTokenProbs = [3][11]uint8{
	{128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	{180, 96, 224, 140, 155, 165, 145, 170, 150, 140, 135},
	{220, 130, 240, 160, 175, 180, 160, 185, 165, 150, 140},
}

// Vp8FrameHeader holds a parsed IETF RFC 6386 Section 9.1 VP8 uncompressed frame header.
type Vp8FrameHeader struct {
	IsKeyframe         bool
	Version            uint8
	ShowFrame          bool
	FirstPartitionSize uint32
	Width              uint16
	Height             uint16
	HorizontalScale    uint8
	VerticalScale      uint8
}

func ParseVp8FrameHeader(data []byte) (Vp8FrameHeader, error) {
	if len(data) < 3 {
		return Vp8FrameHeader{}, &VideoError{Kind: ErrInvalidHeader}
	}
	tag := uint32(data[0]) | (uint32(data[1]) << 8) | (uint32(data[2]) << 16)
	isKeyframe := (tag & 1) == 0
	version := uint8((tag >> 1) & 0x07)
	showFrame := ((tag >> 4) & 1) != 0
	firstPartitionSize := tag >> 5
	if isKeyframe {
		if len(data) < 10 || !bytes.Equal(data[3:6], []byte{0x9D, 0x01, 0x2A}) {
			return Vp8FrameHeader{}, &VideoError{Kind: ErrInvalidHeader}
		}
		rawW := binary.LittleEndian.Uint16(data[6:8])
		rawH := binary.LittleEndian.Uint16(data[8:10])
		return Vp8FrameHeader{
			IsKeyframe:         true,
			Version:            version,
			ShowFrame:          showFrame,
			FirstPartitionSize: firstPartitionSize,
			Width:              rawW & 0x3FFF,
			Height:             rawH & 0x3FFF,
			HorizontalScale:    uint8(rawW >> 14),
			VerticalScale:      uint8(rawH >> 14),
		}, nil
	}
	return Vp8FrameHeader{
		IsKeyframe:         false,
		Version:            version,
		ShowFrame:          showFrame,
		FirstPartitionSize: firstPartitionSize,
	}, nil
}

// Vp8MbIntraMode identifies 16x16 luma / 8x8 chroma intra prediction modes (RFC 6386 Section 12.2).
type Vp8MbIntraMode uint8

const (
	Vp8DcPred Vp8MbIntraMode = iota
	Vp8VPred
	Vp8HPred
	Vp8TmPred
	Vp8BPred
)

// Vp8SubblockMode identifies 4x4 subblock intra prediction modes (RFC 6386 Section 12.3).
type Vp8SubblockMode uint8

const (
	Vp8BDcPred Vp8SubblockMode = iota
	Vp8BTmPred
	Vp8BVePred
	Vp8BHePred
	Vp8BLdPred
	Vp8BRdPred
	Vp8BVrPred
	Vp8BVlPred
	Vp8BHdPred
	Vp8BHuPred
)

func Vp8SubblockModeFromU8(v uint8) Vp8SubblockMode {
	if v > 9 {
		return Vp8BHuPred
	}
	return Vp8SubblockMode(v)
}

// Vp8BoolEncoder implements the IETF RFC 6386 Section 7.2 boolean arithmetic encoder.
type Vp8BoolEncoder struct {
	buffer   []byte
	lowvalue uint64
	rng      uint32
	count    int32
}

func NewVp8BoolEncoder() *Vp8BoolEncoder {
	return &Vp8BoolEncoder{
		buffer:   make([]byte, 0, 256),
		lowvalue: 0,
		rng:      255,
		count:    -24,
	}
}

func (e *Vp8BoolEncoder) WriteBool(bit bool, prob uint8) {
	split := 1 + (((e.rng - 1) * uint32(prob)) >> 8)
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

func (e *Vp8BoolEncoder) WriteLiteral(value uint32, numBits uint8) {
	for shift := int(numBits) - 1; shift >= 0; shift-- {
		bit := ((value >> uint(shift)) & 1) != 0
		e.WriteBool(bit, 128)
	}
}

func (e *Vp8BoolEncoder) WriteSignedLiteral(value int32, numBits uint8) {
	mag := value
	if mag < 0 {
		mag = -mag
	}
	e.WriteLiteral(uint32(mag), numBits)
	e.WriteBool(value < 0, 128)
}

func (e *Vp8BoolEncoder) Finish() []byte {
	for i := 0; i < 32; i++ {
		e.WriteBool(false, 128)
	}
	return e.buffer
}

// Vp8BoolDecoder implements the IETF RFC 6386 Section 7 boolean arithmetic decoder.
type Vp8BoolDecoder struct {
	data     []byte
	pos      int
	rng      uint32
	value    uint32
	bitCount int32
}

func NewVp8BoolDecoder(data []byte) *Vp8BoolDecoder {
	var b0, b1 uint32
	if len(data) > 0 {
		b0 = uint32(data[0])
	}
	if len(data) > 1 {
		b1 = uint32(data[1])
	}
	pos := 2
	if len(data) < 2 {
		pos = len(data)
	}
	return &Vp8BoolDecoder{
		data:     data,
		pos:      pos,
		rng:      255,
		value:    (b0 << 8) | b1,
		bitCount: 0,
	}
}

func (d *Vp8BoolDecoder) ReadBool(prob uint8) bool {
	split := 1 + (((d.rng - 1) * uint32(prob)) >> 8)
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
			if d.pos < len(d.data) {
				d.value |= uint32(d.data[d.pos])
				d.pos++
			}
		}
	}
	return bit
}

func (d *Vp8BoolDecoder) ReadLiteral(numBits uint8) uint32 {
	var v uint32
	for i := uint8(0); i < numBits; i++ {
		v <<= 1
		if d.ReadBool(128) {
			v |= 1
		}
	}
	return v
}

// Vp8QuantFactors holds RFC 6386 Section 14.1 dequantization factors.
type Vp8QuantFactors struct {
	Y1Dc int16
	Y1Ac int16
	Y2Dc int16
	Y2Ac int16
	UvDc int16
	UvAc int16
}

func Vp8QuantFactorsFromQi(yAcQi uint8, yDcDelta, y2DcDelta, y2AcDelta, uvDcDelta, uvAcDelta int8) Vp8QuantFactors {
	clampQi := func(q int32) int {
		return int(clampI32(q, 0, 127))
	}
	base := int32(yAcQi)
	y1Dc := Vp8DcQlookup[clampQi(base+int32(yDcDelta))]
	y1Ac := Vp8AcQlookup[clampQi(base)]
	y2Dc := Vp8DcQlookup[clampQi(base+int32(y2DcDelta))] * 2
	y2Ac := (Vp8AcQlookup[clampQi(base+int32(y2AcDelta))] * 155) / 100
	if y2Ac < 8 {
		y2Ac = 8
	}
	uvDc := Vp8DcQlookup[clampQi(base+int32(uvDcDelta))]
	if uvDc > 132 {
		uvDc = 132
	}
	uvAc := Vp8AcQlookup[clampQi(base+int32(uvAcDelta))]
	return Vp8QuantFactors{
		Y1Dc: y1Dc,
		Y1Ac: y1Ac,
		Y2Dc: y2Dc,
		Y2Ac: y2Ac,
		UvDc: uvDc,
		UvAc: uvAc,
	}
}

// Vp8ForwardDct4x4 computes the forward 4x4 DCT matching Vp8InverseIdct4x4.
func Vp8ForwardDct4x4(input *[16]int16) [16]int32 {
	var tmp [16]int32
	for i := 0; i < 4; i++ {
		row := i * 4
		a1 := (int32(input[row]) + int32(input[row+3])) * 8
		b1 := (int32(input[row+1]) + int32(input[row+2])) * 8
		c1 := (int32(input[row+1]) - int32(input[row+2])) * 8
		d1 := (int32(input[row]) - int32(input[row+3])) * 8
		tmp[row] = a1 + b1
		tmp[row+2] = a1 - b1
		tmp[row+1] = (c1*2217 + d1*5352 + 14500) >> 12
		tmp[row+3] = (d1*2217 - c1*5352 + 7500) >> 12
	}
	var out [16]int32
	for i := 0; i < 4; i++ {
		a1 := tmp[i] + tmp[12+i]
		b1 := tmp[4+i] + tmp[8+i]
		c1 := tmp[4+i] - tmp[8+i]
		d1 := tmp[i] - tmp[12+i]
		out[i] = (a1 + b1 + 7) >> 4
		out[8+i] = (a1 - b1 + 7) >> 4
		nz := int32(0)
		if d1 != 0 {
			nz = 1
		}
		out[4+i] = ((c1*2217 + d1*5352 + 12000) >> 16) + nz
		out[12+i] = (d1*2217 - c1*5352 + 51000) >> 16
	}
	return out
}

// Vp8InverseIdct4x4 computes the RFC 6386 Section 14.3 4x4 inverse DCT.
func Vp8InverseIdct4x4(input *[16]int32) [16]int16 {
	const cospi8Sqrt2Minus1 = 20091
	const sinpi8Sqrt2 = 35468
	var tmp [16]int32
	for i := 0; i < 4; i++ {
		a1 := input[i] + input[8+i]
		b1 := input[i] - input[8+i]
		t1 := (input[4+i] * sinpi8Sqrt2) >> 16
		t2 := input[12+i] + ((input[12+i] * cospi8Sqrt2Minus1) >> 16)
		c1 := t1 - t2
		t1 = input[4+i] + ((input[4+i] * cospi8Sqrt2Minus1) >> 16)
		t2 = (input[12+i] * sinpi8Sqrt2) >> 16
		d1 := t1 + t2
		tmp[i*4] = a1 + d1
		tmp[i*4+3] = a1 - d1
		tmp[i*4+1] = b1 + c1
		tmp[i*4+2] = b1 - c1
	}
	var out [16]int16
	for i := 0; i < 4; i++ {
		a1 := tmp[i] + tmp[8+i]
		b1 := tmp[i] - tmp[8+i]
		t1 := (tmp[4+i] * sinpi8Sqrt2) >> 16
		t2 := tmp[12+i] + ((tmp[12+i] * cospi8Sqrt2Minus1) >> 16)
		c1 := t1 - t2
		t1 = tmp[4+i] + ((tmp[4+i] * cospi8Sqrt2Minus1) >> 16)
		t2 = (tmp[12+i] * sinpi8Sqrt2) >> 16
		d1 := t1 + t2
		out[i*4] = int16(clampI32((a1+d1+4)>>3, -255, 255))
		out[i*4+3] = int16(clampI32((a1-d1+4)>>3, -255, 255))
		out[i*4+1] = int16(clampI32((b1+c1+4)>>3, -255, 255))
		out[i*4+2] = int16(clampI32((b1-c1+4)>>3, -255, 255))
	}
	return out
}

// Vp8ForwardWht4x4 computes the forward 4x4 Walsh-Hadamard Transform matching Vp8InverseWht4x4.
func Vp8ForwardWht4x4(input *[16]int32) [16]int32 {
	var tmp [16]int32
	for i := 0; i < 4; i++ {
		base := i * 4
		a1 := input[base] + input[base+3]
		b1 := input[base+1] + input[base+2]
		c1 := input[base+1] - input[base+2]
		d1 := input[base] - input[base+3]
		tmp[base] = a1 + b1
		tmp[base+1] = c1 + d1
		tmp[base+2] = a1 - b1
		tmp[base+3] = d1 - c1
	}
	var out [16]int32
	for i := 0; i < 4; i++ {
		a1 := tmp[i] + tmp[12+i]
		b1 := tmp[4+i] + tmp[8+i]
		c1 := tmp[4+i] - tmp[8+i]
		d1 := tmp[i] - tmp[12+i]
		out[i] = (a1 + b1 + 1) >> 1
		out[4+i] = (c1 + d1 + 1) >> 1
		out[8+i] = (a1 - b1 + 1) >> 1
		out[12+i] = (d1 - c1 + 1) >> 1
	}
	return out
}

// Vp8InverseWht4x4 computes the RFC 6386 Section 14.3 4x4 inverse Walsh-Hadamard Transform.
func Vp8InverseWht4x4(input *[16]int32) [16]int32 {
	var tmp [16]int32
	for i := 0; i < 4; i++ {
		a1 := input[i] + input[12+i]
		b1 := input[4+i] + input[8+i]
		c1 := input[4+i] - input[8+i]
		d1 := input[i] - input[12+i]
		tmp[i] = a1 + b1
		tmp[4+i] = c1 + d1
		tmp[8+i] = a1 - b1
		tmp[12+i] = d1 - c1
	}
	var out [16]int32
	for i := 0; i < 4; i++ {
		base := i * 4
		a1 := tmp[base] + tmp[base+3]
		b1 := tmp[base+1] + tmp[base+2]
		c1 := tmp[base+1] - tmp[base+2]
		d1 := tmp[base] - tmp[base+3]
		out[base] = (a1 + b1 + 3) >> 3
		out[base+1] = (c1 + d1 + 3) >> 3
		out[base+2] = (a1 - b1 + 3) >> 3
		out[base+3] = (d1 - c1 + 3) >> 3
	}
	return out
}

// Vp8LoopFilterEdge applies the RFC 6386 Section 15 simple loop filter across a 4-sample edge.
func Vp8LoopFilterEdge(p1, p0, q0, q1 byte, limit int32) (byte, byte) {
	p1s := int32(p1) - 128
	p0s := int32(p0) - 128
	q0s := int32(q0) - 128
	q1s := int32(q1) - 128
	d0 := p0s - q0s
	if d0 < 0 {
		d0 = -d0
	}
	d1 := p1s - q1s
	if d1 < 0 {
		d1 = -d1
	}
	if d0*2+d1/2 > limit {
		return p0, q0
	}
	filterVal := clampI32((p1s-q1s)+3*(q0s-p0s), -128, 127)
	f1 := filterVal + 4
	if f1 > 127 {
		f1 = 127
	}
	f1 >>= 3
	f2 := filterVal + 3
	if f2 > 127 {
		f2 = 127
	}
	f2 >>= 3
	newP0 := byte(clampI32(p0s+f2+128, 0, 255))
	newQ0 := byte(clampI32(q0s-f1+128, 0, 255))
	return newP0, newQ0
}

// Vp8PredictIntraMb predicts a 16x16 luma block or 8x8 chroma block using RFC 6386 Section 12.2.
func Vp8PredictIntraMb(mode Vp8MbIntraMode, n int, above, left []byte, topLeft byte, dst []byte, stride int) {
	def16 := [16]byte{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128}
	switch mode {
	case Vp8VPred:
		row := above
		if len(row) < n {
			row = def16[:]
		}
		for y := 0; y < n; y++ {
			copy(dst[y*stride:y*stride+n], row[:n])
		}
	case Vp8HPred:
		col := left
		if len(col) < n {
			col = def16[:]
		}
		for y := 0; y < n; y++ {
			v := col[y]
			for x := 0; x < n; x++ {
				dst[y*stride+x] = v
			}
		}
	case Vp8TmPred:
		row := above
		if len(row) < n {
			row = def16[:]
		}
		col := left
		if len(col) < n {
			col = def16[:]
		}
		p := int32(topLeft)
		for y := 0; y < n; y++ {
			l := int32(col[y])
			for x := 0; x < n; x++ {
				a := int32(row[x])
				dst[y*stride+x] = byte(clampI32(l+a-p, 0, 255))
			}
		}
	default:
		var dc byte = 128
		hasA := len(above) >= n
		hasL := len(left) >= n
		switch {
		case hasA && hasL:
			var sum uint32
			for i := 0; i < n; i++ {
				sum += uint32(above[i]) + uint32(left[i])
			}
			dc = byte((sum + uint32(n)) / uint32(2*n))
		case hasA:
			var sum uint32
			for i := 0; i < n; i++ {
				sum += uint32(above[i])
			}
			dc = byte((sum + uint32(n/2)) / uint32(n))
		case hasL:
			var sum uint32
			for i := 0; i < n; i++ {
				sum += uint32(left[i])
			}
			dc = byte((sum + uint32(n/2)) / uint32(n))
		}
		for y := 0; y < n; y++ {
			for x := 0; x < n; x++ {
				dst[y*stride+x] = dc
			}
		}
	}
}

// Vp8PredictIntra4x4 predicts a 4x4 luma subblock using RFC 6386 Section 12.3.
func Vp8PredictIntra4x4(mode Vp8SubblockMode, above *[8]byte, left *[4]byte, topLeft byte) [16]byte {
	avg2 := func(a, b byte) byte { return byte((uint16(a) + uint16(b) + 1) >> 1) }
	avg3 := func(a, b, c byte) byte { return byte((uint16(a) + 2*uint16(b) + uint16(c) + 2) >> 2) }
	var out [16]byte
	for i := range out {
		out[i] = 128
	}
	switch mode {
	case Vp8BDcPred:
		var sum uint32
		for i := 0; i < 4; i++ {
			sum += uint32(above[i]) + uint32(left[i])
		}
		dc := byte((sum + 4) >> 3)
		for i := range out {
			out[i] = dc
		}
	case Vp8BTmPred:
		p := int32(topLeft)
		for r := 0; r < 4; r++ {
			for c := 0; c < 4; c++ {
				out[r*4+c] = byte(clampI32(int32(left[r])+int32(above[c])-p, 0, 255))
			}
		}
	case Vp8BVePred:
		row := [4]byte{
			avg3(topLeft, above[0], above[1]),
			avg3(above[0], above[1], above[2]),
			avg3(above[1], above[2], above[3]),
			avg3(above[2], above[3], above[4]),
		}
		for r := 0; r < 4; r++ {
			copy(out[r*4:r*4+4], row[:])
		}
	case Vp8BHePred:
		col := [4]byte{
			avg3(topLeft, left[0], left[1]),
			avg3(left[0], left[1], left[2]),
			avg3(left[1], left[2], left[3]),
			avg3(left[2], left[3], left[3]),
		}
		for r := 0; r < 4; r++ {
			for c := 0; c < 4; c++ {
				out[r*4+c] = col[r]
			}
		}
	case Vp8BLdPred:
		for r := 0; r < 4; r++ {
			for c := 0; c < 4; c++ {
				i := r + c
				if i == 6 {
					out[r*4+c] = avg3(above[6], above[7], above[7])
				} else {
					out[r*4+c] = avg3(above[i], above[i+1], above[i+2])
				}
			}
		}
	case Vp8BRdPred:
		e := [9]byte{left[3], left[2], left[1], left[0], topLeft, above[0], above[1], above[2], above[3]}
		for r := 0; r < 4; r++ {
			for c := 0; c < 4; c++ {
				idx := 4 + c - r
				out[r*4+c] = avg3(e[idx-1], e[idx], e[idx+1])
			}
		}
	case Vp8BVrPred:
		e := [9]byte{left[3], left[2], left[1], left[0], topLeft, above[0], above[1], above[2], above[3]}
		for r := 0; r < 4; r++ {
			for c := 0; c < 4; c++ {
				idx := c - (r >> 1)
				if (r & 1) == 0 {
					base := 4 + idx
					out[r*4+c] = avg2(e[base], e[base+1])
				} else {
					base := 3 + idx
					if base < 1 {
						base = 1
					}
					out[r*4+c] = avg3(e[base-1], e[base], e[base+1])
				}
			}
		}
	case Vp8BVlPred:
		for r := 0; r < 4; r++ {
			for c := 0; c < 4; c++ {
				base := c + (r >> 1)
				if (r & 1) == 0 {
					out[r*4+c] = avg2(above[base], above[base+1])
				} else {
					out[r*4+c] = avg3(above[base], above[base+1], above[base+2])
				}
			}
		}
	case Vp8BHdPred:
		e := [9]byte{left[3], left[2], left[1], left[0], topLeft, above[0], above[1], above[2], above[3]}
		for r := 0; r < 4; r++ {
			for c := 0; c < 4; c++ {
				idx := int(clampI32(int32(4+r-(c>>1)), 1, 7))
				if (c & 1) == 0 {
					out[r*4+c] = avg2(e[8-idx], e[9-idx])
				} else {
					out[r*4+c] = avg3(e[7-idx], e[8-idx], e[9-idx])
				}
			}
		}
	default:
		for r := 0; r < 4; r++ {
			for c := 0; c < 4; c++ {
				z := 2*r + c
				switch z {
				case 0:
					out[r*4+c] = avg2(left[0], left[1])
				case 1:
					out[r*4+c] = avg3(left[0], left[1], left[2])
				case 2:
					out[r*4+c] = avg2(left[1], left[2])
				case 3:
					out[r*4+c] = avg3(left[1], left[2], left[3])
				case 4:
					out[r*4+c] = avg2(left[2], left[3])
				case 5:
					out[r*4+c] = avg3(left[2], left[3], left[3])
				default:
					out[r*4+c] = left[3]
				}
			}
		}
	}
	return out
}

// Vp8EncodeCoeffBlock encodes a 4x4 block of quantized coefficients using the RFC 6386 Section 13.2 token tree.
func Vp8EncodeCoeffBlock(enc *Vp8BoolEncoder, coeffs *[16]int16, firstCoeff, initialCtx int) bool {
	endIdx := -1
	for i := firstCoeff; i < 16; i++ {
		if coeffs[Vp8Zigzag[i]] != 0 {
			endIdx = i
		}
	}
	ctx := initialCtx
	if ctx > 2 {
		ctx = 2
	}
	if endIdx < 0 {
		p := DefaultTokenProbs[ctx]
		enc.WriteBool(false, p[0])
		return false
	}
	for i := firstCoeff; i < 16; i++ {
		p := DefaultTokenProbs[ctx]
		if i > endIdx {
			enc.WriteBool(false, p[0])
			break
		}
		val := int32(coeffs[Vp8Zigzag[i]])
		absV := val
		if absV < 0 {
			absV = -absV
		}
		enc.WriteBool(true, p[0])
		if absV == 0 {
			enc.WriteBool(false, p[1])
			ctx = 0
			continue
		}
		enc.WriteBool(true, p[1])
		if absV == 1 {
			enc.WriteBool(false, p[2])
			ctx = 1
		} else {
			enc.WriteBool(true, p[2])
			if absV <= 4 {
				enc.WriteBool(false, p[3])
				if absV == 2 {
					enc.WriteBool(false, p[4])
				} else {
					enc.WriteBool(true, p[4])
					enc.WriteBool(absV == 4, p[5])
				}
			} else {
				enc.WriteBool(true, p[3])
				if absV <= 10 {
					enc.WriteBool(false, p[6])
					if absV <= 6 {
						enc.WriteBool(false, p[7])
						extra := absV - 5
						enc.WriteBool((extra&1) != 0, Pcat1[0])
					} else {
						enc.WriteBool(true, p[7])
						extra := absV - 7
						enc.WriteBool(((extra>>1)&1) != 0, Pcat2[0])
						enc.WriteBool((extra&1) != 0, Pcat2[1])
					}
				} else {
					enc.WriteBool(true, p[6])
					if absV <= 34 {
						enc.WriteBool(false, p[8])
						if absV <= 18 {
							enc.WriteBool(false, p[9])
							extra := absV - 11
							for b := 0; b < 3; b++ {
								enc.WriteBool(((extra>>uint(2-b))&1) != 0, Pcat3[b])
							}
						} else {
							enc.WriteBool(true, p[9])
							extra := absV - 19
							for b := 0; b < 4; b++ {
								enc.WriteBool(((extra>>uint(3-b))&1) != 0, Pcat4[b])
							}
						}
					} else {
						enc.WriteBool(true, p[8])
						if absV <= 66 {
							enc.WriteBool(false, p[10])
							extra := absV - 35
							for b := 0; b < 5; b++ {
								enc.WriteBool(((extra>>uint(4-b))&1) != 0, Pcat5[b])
							}
						} else {
							enc.WriteBool(true, p[10])
							extra := absV - 67
							if extra > 1981 {
								extra = 1981
							}
							for b := 0; b < 11; b++ {
								enc.WriteBool(((extra>>uint(10-b))&1) != 0, Pcat6[b])
							}
						}
					}
				}
			}
			ctx = 2
		}
		enc.WriteBool(val < 0, 128)
	}
	return true
}

// Vp8DecodeCoeffBlock decodes a 4x4 block of quantized coefficients using the RFC 6386 Section 13.2 token tree.
func Vp8DecodeCoeffBlock(dec *Vp8BoolDecoder, firstCoeff, initialCtx int, dcQ, acQ int16) ([16]int32, bool) {
	var out [16]int32
	ctx := initialCtx
	if ctx > 2 {
		ctx = 2
	}
	hasNz := false
	for i := firstCoeff; i < 16; i++ {
		p := DefaultTokenProbs[ctx]
		if !dec.ReadBool(p[0]) {
			break
		}
		if !dec.ReadBool(p[1]) {
			ctx = 0
			continue
		}
		var absV int32
		if !dec.ReadBool(p[2]) {
			ctx = 1
			absV = 1
		} else {
			ctx = 2
			if !dec.ReadBool(p[3]) {
				if !dec.ReadBool(p[4]) {
					absV = 2
				} else if !dec.ReadBool(p[5]) {
					absV = 3
				} else {
					absV = 4
				}
			} else if !dec.ReadBool(p[6]) {
				if !dec.ReadBool(p[7]) {
					absV = 5
					if dec.ReadBool(Pcat1[0]) {
						absV++
					}
				} else {
					var b0, b1 int32
					if dec.ReadBool(Pcat2[0]) {
						b0 = 1
					}
					if dec.ReadBool(Pcat2[1]) {
						b1 = 1
					}
					absV = 7 + (b0 << 1) + b1
				}
			} else if !dec.ReadBool(p[8]) {
				if !dec.ReadBool(p[9]) {
					var extra int32
					for _, prob := range Pcat3 {
						extra <<= 1
						if dec.ReadBool(prob) {
							extra |= 1
						}
					}
					absV = 11 + extra
				} else {
					var extra int32
					for _, prob := range Pcat4 {
						extra <<= 1
						if dec.ReadBool(prob) {
							extra |= 1
						}
					}
					absV = 19 + extra
				}
			} else if !dec.ReadBool(p[10]) {
				var extra int32
				for _, prob := range Pcat5 {
					extra <<= 1
					if dec.ReadBool(prob) {
						extra |= 1
					}
				}
				absV = 35 + extra
			} else {
				var extra int32
				for _, prob := range Pcat6 {
					extra <<= 1
					if dec.ReadBool(prob) {
						extra |= 1
					}
				}
				absV = 67 + extra
			}
		}
		sign := dec.ReadBool(128)
		signed := absV
		if sign {
			signed = -absV
		}
		zz := Vp8Zigzag[i]
		q := int32(acQ)
		if zz == 0 {
			q = int32(dcQ)
		}
		out[zz] = signed * q
		hasNz = true
	}
	return out, hasNz
}

func chooseBestIntraMode(src []byte, srcStride, n int, above, left []byte, topLeft byte) (Vp8MbIntraMode, []byte) {
	modes := [4]Vp8MbIntraMode{Vp8DcPred, Vp8VPred, Vp8HPred, Vp8TmPred}
	bestMode := Vp8DcPred
	bestPred := make([]byte, n*n)
	for i := range bestPred {
		bestPred[i] = 128
	}
	bestSad := uint64(math.MaxUint64)
	for _, mode := range modes {
		if mode == Vp8VPred && len(above) == 0 {
			continue
		}
		if mode == Vp8HPred && len(left) == 0 {
			continue
		}
		if mode == Vp8TmPred && (len(above) == 0 || len(left) == 0) {
			continue
		}
		pred := make([]byte, n*n)
		Vp8PredictIntraMb(mode, n, above, left, topLeft, pred, n)
		var sad uint64
		for y := 0; y < n; y++ {
			for x := 0; x < n; x++ {
				diff := int32(src[y*srcStride+x]) - int32(pred[y*n+x])
				if diff < 0 {
					diff = -diff
				}
				sad += uint64(diff)
			}
		}
		if sad < bestSad {
			bestSad = sad
			bestMode = mode
			bestPred = pred
		}
	}
	return bestMode, bestPred
}

func writeKfYmode(enc *Vp8BoolEncoder, mode Vp8MbIntraMode) {
	if mode == Vp8BPred {
		enc.WriteBool(false, KfYmodeProbs[0])
		return
	}
	enc.WriteBool(true, KfYmodeProbs[0])
	switch mode {
	case Vp8DcPred:
		enc.WriteBool(false, KfYmodeProbs[1])
		enc.WriteBool(false, KfYmodeProbs[2])
	case Vp8VPred:
		enc.WriteBool(false, KfYmodeProbs[1])
		enc.WriteBool(true, KfYmodeProbs[2])
	case Vp8HPred:
		enc.WriteBool(true, KfYmodeProbs[1])
		enc.WriteBool(false, KfYmodeProbs[3])
	default:
		enc.WriteBool(true, KfYmodeProbs[1])
		enc.WriteBool(true, KfYmodeProbs[3])
	}
}

func readKfYmode(dec *Vp8BoolDecoder) Vp8MbIntraMode {
	if !dec.ReadBool(KfYmodeProbs[0]) {
		return Vp8BPred
	}
	if !dec.ReadBool(KfYmodeProbs[1]) {
		if !dec.ReadBool(KfYmodeProbs[2]) {
			return Vp8DcPred
		}
		return Vp8VPred
	}
	if !dec.ReadBool(KfYmodeProbs[3]) {
		return Vp8HPred
	}
	return Vp8TmPred
}

func writeKfUvMode(enc *Vp8BoolEncoder, mode Vp8MbIntraMode) {
	switch mode {
	case Vp8DcPred:
		enc.WriteBool(false, KfUvModeProbs[0])
	case Vp8VPred:
		enc.WriteBool(true, KfUvModeProbs[0])
		enc.WriteBool(false, KfUvModeProbs[1])
	case Vp8HPred:
		enc.WriteBool(true, KfUvModeProbs[0])
		enc.WriteBool(true, KfUvModeProbs[1])
		enc.WriteBool(false, KfUvModeProbs[2])
	default:
		enc.WriteBool(true, KfUvModeProbs[0])
		enc.WriteBool(true, KfUvModeProbs[1])
		enc.WriteBool(true, KfUvModeProbs[2])
	}
}

func readKfUvMode(dec *Vp8BoolDecoder) Vp8MbIntraMode {
	if !dec.ReadBool(KfUvModeProbs[0]) {
		return Vp8DcPred
	}
	if !dec.ReadBool(KfUvModeProbs[1]) {
		return Vp8VPred
	}
	if !dec.ReadBool(KfUvModeProbs[2]) {
		return Vp8HPred
	}
	return Vp8TmPred
}

func applyVp8SimpleLoopFilter(yPlane []byte, stride, height int, level uint8) {
	limit := int32(level)*2 + 4
	for x := 16; x < stride; x += 16 {
		for y := 0; y < height; y++ {
			row := y * stride
			np0, nq0 := Vp8LoopFilterEdge(yPlane[row+x-2], yPlane[row+x-1], yPlane[row+x], yPlane[row+x+1], limit)
			yPlane[row+x-1] = np0
			yPlane[row+x] = nq0
		}
	}
	for y := 16; y < height; y += 16 {
		for x := 0; x < stride; x++ {
			np0, nq0 := Vp8LoopFilterEdge(yPlane[(y-2)*stride+x], yPlane[(y-1)*stride+x], yPlane[y*stride+x], yPlane[(y+1)*stride+x], limit)
			yPlane[(y-1)*stride+x] = np0
			yPlane[y*stride+x] = nq0
		}
	}
}

// EncodeVp8Rfc6386Frame encodes a single VideoFrame into an RFC 6386 VP8 frame.
func EncodeVp8Rfc6386Frame(frame *VideoFrame, qp uint8, isKeyframe bool, prevY, prevU, prevV *[]byte) ([]byte, error) {
	if err := ValidateVideoDimensions(frame.Width, frame.Height); err != nil {
		return nil, err
	}
	w, h := int(frame.Width), int(frame.Height)
	mbCols := (w + 15) / 16
	mbRows := (h + 15) / 16
	padW := mbCols * 16
	padH := mbRows * 16
	uvW := mbCols * 8
	uvH := mbRows * 8

	i420, ok := I420FromRgbaBt709(frame.Width, frame.Height, frame.Rgba)
	if !ok {
		return nil, &VideoError{Kind: ErrInvalidHeader}
	}
	srcY := make([]byte, padW*padH)
	srcU := make([]byte, uvW*uvH)
	srcV := make([]byte, uvW*uvH)
	for py := 0; py < padH; py++ {
		sy := py
		if sy >= h {
			sy = h - 1
		}
		for px := 0; px < padW; px++ {
			sx := px
			if sx >= w {
				sx = w - 1
			}
			srcY[py*padW+px] = i420.YPlane[sy*w+sx]
			if (px&1) == 0 && (py&1) == 0 {
				cw, ch := w/2, h/2
				csx, csy := sx/2, sy/2
				if csx >= cw {
					csx = cw - 1
				}
				if csy >= ch {
					csy = ch - 1
				}
				srcU[(py/2)*uvW+(px/2)] = i420.UPlane[csy*cw+csx]
				srcV[(py/2)*uvW+(px/2)] = i420.VPlane[csy*cw+csx]
			}
		}
	}
	qi := uint8(clampI32(int32(qp), 1, 127))
	qf := Vp8QuantFactorsFromQi(qi, 0, 0, 0, 0, 0)
	loopFilterLevel := uint8(clampI32(int32(qi/4), 0, 63))

	p0 := NewVp8BoolEncoder()
	p1 := NewVp8BoolEncoder()

	useInter := !isKeyframe && len(*prevY) == padW*padH
	if !useInter {
		p0.WriteBool(false, 128)
		p0.WriteBool(false, 128)
	}
	p0.WriteBool(false, 128)
	p0.WriteBool(true, 128)
	p0.WriteLiteral(uint32(loopFilterLevel), 6)
	p0.WriteLiteral(0, 3)
	p0.WriteBool(false, 128)
	p0.WriteLiteral(0, 2)
	p0.WriteLiteral(uint32(qi), 7)
	for i := 0; i < 5; i++ {
		p0.WriteBool(false, 128)
	}
	if useInter {
		p0.WriteBool(false, 128)
		p0.WriteBool(false, 128)
		p0.WriteLiteral(0, 2)
		p0.WriteLiteral(0, 2)
		p0.WriteBool(false, 128)
		p0.WriteBool(false, 128)
		p0.WriteBool(false, 128)
		p0.WriteBool(true, 128)
	} else {
		p0.WriteBool(false, 128)
	}
	for i := 0; i < 4*8*3*11; i++ {
		p0.WriteBool(false, 252)
	}
	p0.WriteBool(true, 128)
	p0.WriteLiteral(128, 8)
	if useInter {
		p0.WriteLiteral(63, 8)
		p0.WriteLiteral(255, 8)
		p0.WriteLiteral(128, 8)
		p0.WriteBool(false, 128)
		p0.WriteBool(false, 128)
		for i := 0; i < 2*19; i++ {
			p0.WriteBool(false, 252)
		}
	}

	reconY := make([]byte, padW*padH)
	reconU := make([]byte, uvW*uvH)
	reconV := make([]byte, uvW*uvH)
	for i := range reconY {
		reconY[i] = 128
	}
	for i := range reconU {
		reconU[i] = 128
		reconV[i] = 128
	}

	for mbr := 0; mbr < mbRows; mbr++ {
		for mbc := 0; mbc < mbCols; mbc++ {
			yX0 := mbc * 16
			yY0 := mbr * 16
			uvX0 := mbc * 8
			uvY0 := mbr * 8

			var aboveY, leftY []byte
			if mbr > 0 {
				aboveY = make([]byte, 16)
				copy(aboveY, reconY[(yY0-1)*padW+yX0:(yY0-1)*padW+yX0+16])
			}
			if mbc > 0 {
				leftY = make([]byte, 16)
				for r := 0; r < 16; r++ {
					leftY[r] = reconY[(yY0+r)*padW+(yX0-1)]
				}
			}
			tlY := byte(129)
			if mbr > 0 && mbc > 0 {
				tlY = reconY[(yY0-1)*padW+(yX0-1)]
			} else if mbr > 0 {
				tlY = 127
			}

			var aboveU, leftU, aboveV, leftV []byte
			if mbr > 0 {
				aboveU = make([]byte, 8)
				aboveV = make([]byte, 8)
				copy(aboveU, reconU[(uvY0-1)*uvW+uvX0:(uvY0-1)*uvW+uvX0+8])
				copy(aboveV, reconV[(uvY0-1)*uvW+uvX0:(uvY0-1)*uvW+uvX0+8])
			}
			if mbc > 0 {
				leftU = make([]byte, 8)
				leftV = make([]byte, 8)
				for r := 0; r < 8; r++ {
					leftU[r] = reconU[(uvY0+r)*uvW+(uvX0-1)]
					leftV[r] = reconV[(uvY0+r)*uvW+(uvX0-1)]
				}
			}
			tlU, tlV := byte(128), byte(128)
			if mbr > 0 && mbc > 0 {
				tlU = reconU[(uvY0-1)*uvW+(uvX0-1)]
				tlV = reconV[(uvY0-1)*uvW+(uvX0-1)]
			}

			ymode, predY := chooseBestIntraMode(srcY[yY0*padW+yX0:], padW, 16, aboveY, leftY, tlY)
			uvMode, predU := chooseBestIntraMode(srcU[uvY0*uvW+uvX0:], uvW, 8, aboveU, leftU, tlU)
			predV := make([]byte, 64)
			Vp8PredictIntraMb(uvMode, 8, aboveV, leftV, tlV, predV, 8)

			codedInterZeromv := false
			if useInter {
				var intraSad, interSad uint64
				for r := 0; r < 16; r++ {
					for c := 0; c < 16; c++ {
						s := int32(srcY[(yY0+r)*padW+(yX0+c)])
						d1 := s - int32(predY[r*16+c])
						if d1 < 0 {
							d1 = -d1
						}
						d2 := s - int32((*prevY)[(yY0+r)*padW+(yX0+c)])
						if d2 < 0 {
							d2 = -d2
						}
						intraSad += uint64(d1)
						interSad += uint64(d2)
					}
				}
				if interSad <= intraSad {
					codedInterZeromv = true
					for r := 0; r < 16; r++ {
						for c := 0; c < 16; c++ {
							predY[r*16+c] = (*prevY)[(yY0+r)*padW+(yX0+c)]
						}
					}
					for r := 0; r < 8; r++ {
						for c := 0; c < 8; c++ {
							predU[r*8+c] = (*prevU)[(uvY0+r)*uvW+(uvX0+c)]
							predV[r*8+c] = (*prevV)[(uvY0+r)*uvW+(uvX0+c)]
						}
					}
				}
			}

			var yAcQ [16][16]int16
			var y2DcIn [16]int32
			anyNz := false
			for sbR := 0; sbR < 4; sbR++ {
				for sbC := 0; sbC < 4; sbC++ {
					sbIdx := sbR*4 + sbC
					var diff [16]int16
					for r := 0; r < 4; r++ {
						for c := 0; c < 4; c++ {
							py := sbR*4 + r
							px := sbC*4 + c
							diff[r*4+c] = int16(srcY[(yY0+py)*padW+(yX0+px)]) - int16(predY[py*16+px])
						}
					}
					dct := Vp8ForwardDct4x4(&diff)
					y2DcIn[sbIdx] = dct[0]
					for k := 1; k < 16; k++ {
						qv := int16(clampI32(dct[k]/int32(qf.Y1Ac), -2048, 2048))
						yAcQ[sbIdx][k] = qv
						if qv != 0 {
							anyNz = true
						}
					}
				}
			}
			y2Wht := Vp8ForwardWht4x4(&y2DcIn)
			var y2Q [16]int16
			for k := 0; k < 16; k++ {
				qdiv := int32(qf.Y2Ac)
				if k == 0 {
					qdiv = int32(qf.Y2Dc)
				}
				qv := int16(clampI32(y2Wht[k]/qdiv, -2048, 2048))
				y2Q[k] = qv
				if qv != 0 {
					anyNz = true
				}
			}
			var uQ, vQ [4][16]int16
			for sbR := 0; sbR < 2; sbR++ {
				for sbC := 0; sbC < 2; sbC++ {
					sbIdx := sbR*2 + sbC
					var diffU, diffV [16]int16
					for r := 0; r < 4; r++ {
						for c := 0; c < 4; c++ {
							py := sbR*4 + r
							px := sbC*4 + c
							diffU[r*4+c] = int16(srcU[(uvY0+py)*uvW+(uvX0+px)]) - int16(predU[py*8+px])
							diffV[r*4+c] = int16(srcV[(uvY0+py)*uvW+(uvX0+px)]) - int16(predV[py*8+px])
						}
					}
					dctU := Vp8ForwardDct4x4(&diffU)
					dctV := Vp8ForwardDct4x4(&diffV)
					for k := 0; k < 16; k++ {
						qdiv := int32(qf.UvAc)
						if k == 0 {
							qdiv = int32(qf.UvDc)
						}
						qu := int16(clampI32(dctU[k]/qdiv, -2048, 2048))
						qv := int16(clampI32(dctV[k]/qdiv, -2048, 2048))
						uQ[sbIdx][k] = qu
						vQ[sbIdx][k] = qv
						if qu != 0 || qv != 0 {
							anyNz = true
						}
					}
				}
			}

			mbSkip := !anyNz
			p0.WriteBool(mbSkip, 128)
			if useInter {
				p0.WriteBool(codedInterZeromv, 63)
				if codedInterZeromv {
					p0.WriteBool(false, 255)
					p0.WriteBool(false, 128)
				} else {
					writeKfYmode(p0, ymode)
					writeKfUvMode(p0, uvMode)
				}
			} else {
				writeKfYmode(p0, ymode)
				writeKfUvMode(p0, uvMode)
			}

			if !mbSkip {
				y2Nz := Vp8EncodeCoeffBlock(p1, &y2Q, 0, 0)
				prevNz := 0
				if y2Nz {
					prevNz = 1
				}
				for sb := 0; sb < 16; sb++ {
					nz := Vp8EncodeCoeffBlock(p1, &yAcQ[sb], 1, prevNz)
					prevNz = 0
					if nz {
						prevNz = 1
					}
				}
				for sb := 0; sb < 4; sb++ {
					nz := Vp8EncodeCoeffBlock(p1, &uQ[sb], 0, prevNz)
					prevNz = 0
					if nz {
						prevNz = 1
					}
				}
				for sb := 0; sb < 4; sb++ {
					nz := Vp8EncodeCoeffBlock(p1, &vQ[sb], 0, prevNz)
					prevNz = 0
					if nz {
						prevNz = 1
					}
				}
			}

			var y2Deq [16]int32
			if !mbSkip {
				for k := 0; k < 16; k++ {
					qdiv := int32(qf.Y2Ac)
					if k == 0 {
						qdiv = int32(qf.Y2Dc)
					}
					y2Deq[k] = int32(y2Q[k]) * qdiv
				}
			}
			y2DcRec := Vp8InverseWht4x4(&y2Deq)
			for sbR := 0; sbR < 4; sbR++ {
				for sbC := 0; sbC < 4; sbC++ {
					sbIdx := sbR*4 + sbC
					var deq [16]int32
					deq[0] = y2DcRec[sbIdx]
					if !mbSkip {
						for k := 1; k < 16; k++ {
							deq[k] = int32(yAcQ[sbIdx][k]) * int32(qf.Y1Ac)
						}
					}
					res := Vp8InverseIdct4x4(&deq)
					for r := 0; r < 4; r++ {
						for c := 0; c < 4; c++ {
							py := sbR*4 + r
							px := sbC*4 + c
							v := clampI32(int32(predY[py*16+px])+int32(res[r*4+c]), 0, 255)
							reconY[(yY0+py)*padW+(yX0+px)] = byte(v)
						}
					}
				}
			}
			for sbR := 0; sbR < 2; sbR++ {
				for sbC := 0; sbC < 2; sbC++ {
					sbIdx := sbR*2 + sbC
					var deqU, deqV [16]int32
					if !mbSkip {
						for k := 0; k < 16; k++ {
							qdiv := int32(qf.UvAc)
							if k == 0 {
								qdiv = int32(qf.UvDc)
							}
							deqU[k] = int32(uQ[sbIdx][k]) * qdiv
							deqV[k] = int32(vQ[sbIdx][k]) * qdiv
						}
					}
					resU := Vp8InverseIdct4x4(&deqU)
					resV := Vp8InverseIdct4x4(&deqV)
					for r := 0; r < 4; r++ {
						for c := 0; c < 4; c++ {
							py := sbR*4 + r
							px := sbC*4 + c
							reconU[(uvY0+py)*uvW+(uvX0+px)] = byte(clampI32(int32(predU[py*8+px])+int32(resU[r*4+c]), 0, 255))
							reconV[(uvY0+py)*uvW+(uvX0+px)] = byte(clampI32(int32(predV[py*8+px])+int32(resV[r*4+c]), 0, 255))
						}
					}
				}
			}
		}
	}

	if loopFilterLevel > 0 {
		applyVp8SimpleLoopFilter(reconY, padW, padH, loopFilterLevel)
	}
	*prevY = reconY
	*prevU = reconU
	*prevV = reconV

	p0Bytes := p0.Finish()
	p1Bytes := p1.Finish()
	firstPartSize := uint32(len(p0Bytes))
	keyBit := uint32(0)
	if !useInter {
		keyBit = 0
	} else {
		keyBit = 1
	}
	tag := keyBit | (1 << 4) | (firstPartSize << 5)

	var out bytes.Buffer
	out.WriteByte(byte(tag & 0xFF))
	out.WriteByte(byte((tag >> 8) & 0xFF))
	out.WriteByte(byte((tag >> 16) & 0xFF))
	if !useInter {
		out.Write([]byte{0x9D, 0x01, 0x2A})
		var dims [4]byte
		binary.LittleEndian.PutUint16(dims[0:2], uint16(w)&0x3FFF)
		binary.LittleEndian.PutUint16(dims[2:4], uint16(h)&0x3FFF)
		out.Write(dims[:])
	}
	out.Write(p0Bytes)
	out.Write(p1Bytes)
	return out.Bytes(), nil
}

// DecodeVp8Rfc6386Frame decodes a single RFC 6386 VP8 frame into a VideoFrame.
func DecodeVp8Rfc6386Frame(data []byte, curWidth, curHeight *uint32, prevY, prevU, prevV *[]byte, ptsMs uint32) (VideoFrame, error) {
	header, err := ParseVp8FrameHeader(data)
	if err != nil {
		return VideoFrame{}, err
	}
	hdrLen := 3
	if header.IsKeyframe {
		*curWidth = uint32(header.Width)
		*curHeight = uint32(header.Height)
		hdrLen = 10
	}
	if err := ValidateVideoDimensions(*curWidth, *curHeight); err != nil {
		return VideoFrame{}, err
	}
	p0Len := int(header.FirstPartitionSize)
	if hdrLen+p0Len > len(data) {
		return VideoFrame{}, &VideoError{Kind: ErrCorruptedBitstream}
	}
	p0Slice := data[hdrLen : hdrLen+p0Len]
	p1Slice := data[hdrLen+p0Len:]
	p0 := NewVp8BoolDecoder(p0Slice)

	if header.IsKeyframe {
		_ = p0.ReadBool(128)
		_ = p0.ReadBool(128)
	}
	if p0.ReadBool(128) {
		updateMap := p0.ReadBool(128)
		updateData := p0.ReadBool(128)
		if updateData {
			_ = p0.ReadBool(128)
			for i := 0; i < 4; i++ {
				if p0.ReadBool(128) {
					_ = p0.ReadLiteral(8)
				}
			}
			for i := 0; i < 4; i++ {
				if p0.ReadBool(128) {
					_ = p0.ReadLiteral(7)
				}
			}
		}
		if updateMap {
			for i := 0; i < 3; i++ {
				if p0.ReadBool(128) {
					_ = p0.ReadLiteral(8)
				}
			}
		}
	}
	_ = p0.ReadBool(128)
	loopFilterLevel := uint8(p0.ReadLiteral(6))
	_ = p0.ReadLiteral(3)
	lfDeltaEnabled := p0.ReadBool(128)
	if lfDeltaEnabled && p0.ReadBool(128) {
		for i := 0; i < 8; i++ {
			if p0.ReadBool(128) {
				_ = p0.ReadLiteral(7)
			}
		}
	}
	log2Partitions := int(p0.ReadLiteral(2))
	numPartitions := 1 << uint(log2Partitions)
	partTableBytes := (numPartitions - 1) * 3
	if len(p1Slice) < partTableBytes {
		return VideoFrame{}, &VideoError{Kind: ErrCorruptedBitstream}
	}
	p1 := NewVp8BoolDecoder(p1Slice[partTableBytes:])

	yAcQi := uint8(p0.ReadLiteral(7))
	readQDelta := func(dec *Vp8BoolDecoder) int8 {
		if dec.ReadBool(128) {
			mag := int8(dec.ReadLiteral(4))
			if dec.ReadBool(128) {
				return -mag
			}
			return mag
		}
		return 0
	}
	yDcDelta := readQDelta(p0)
	y2DcDelta := readQDelta(p0)
	y2AcDelta := readQDelta(p0)
	uvDcDelta := readQDelta(p0)
	uvAcDelta := readQDelta(p0)
	qf := Vp8QuantFactorsFromQi(yAcQi, yDcDelta, y2DcDelta, y2AcDelta, uvDcDelta, uvAcDelta)

	if !header.IsKeyframe {
		_ = p0.ReadBool(128)
		_ = p0.ReadBool(128)
		_ = p0.ReadLiteral(2)
		_ = p0.ReadLiteral(2)
		_ = p0.ReadBool(128)
		_ = p0.ReadBool(128)
		_ = p0.ReadBool(128)
		_ = p0.ReadBool(128)
	} else {
		_ = p0.ReadBool(128)
	}
	for i := 0; i < 4*8*3*11; i++ {
		if p0.ReadBool(252) {
			_ = p0.ReadLiteral(8)
		}
	}
	mbNoSkipCoeff := p0.ReadBool(128)
	probSkipFalse := uint8(128)
	if mbNoSkipCoeff {
		probSkipFalse = uint8(p0.ReadLiteral(8))
	}
	probIntra := uint8(63)
	probLast := uint8(255)
	if !header.IsKeyframe {
		probIntra = uint8(p0.ReadLiteral(8))
		probLast = uint8(p0.ReadLiteral(8))
		_ = p0.ReadLiteral(8)
		if p0.ReadBool(128) {
			for i := 0; i < 4; i++ {
				_ = p0.ReadLiteral(8)
			}
		}
		if p0.ReadBool(128) {
			for i := 0; i < 3; i++ {
				_ = p0.ReadLiteral(8)
			}
		}
		for i := 0; i < 2*19; i++ {
			if p0.ReadBool(252) {
				_ = p0.ReadLiteral(7)
			}
		}
	}

	w, h := int(*curWidth), int(*curHeight)
	mbCols := (w + 15) / 16
	mbRows := (h + 15) / 16
	padW := mbCols * 16
	padH := mbRows * 16
	uvW := mbCols * 8
	uvH := mbRows * 8

	reconY := make([]byte, padW*padH)
	reconU := make([]byte, uvW*uvH)
	reconV := make([]byte, uvW*uvH)
	for i := range reconY {
		reconY[i] = 128
	}
	for i := range reconU {
		reconU[i] = 128
		reconV[i] = 128
	}

	for mbr := 0; mbr < mbRows; mbr++ {
		for mbc := 0; mbc < mbCols; mbc++ {
			yX0 := mbc * 16
			yY0 := mbr * 16
			uvX0 := mbc * 8
			uvY0 := mbr * 8

			mbSkip := false
			if mbNoSkipCoeff {
				mbSkip = p0.ReadBool(probSkipFalse)
			}
			isInterMb := false
			ymode := Vp8DcPred
			uvMode := Vp8DcPred
			if !header.IsKeyframe {
				isInterMb = p0.ReadBool(probIntra)
				if isInterMb {
					_ = p0.ReadBool(probLast)
					_ = p0.ReadBool(128)
				} else {
					ymode = readKfYmode(p0)
					uvMode = readKfUvMode(p0)
				}
			} else {
				ymode = readKfYmode(p0)
				if ymode == Vp8BPred {
					for i := 0; i < 16; i++ {
						_ = p0.ReadBool(DefaultBmodeProbs[0])
					}
				}
				uvMode = readKfUvMode(p0)
			}

			predY := make([]byte, 256)
			predU := make([]byte, 64)
			predV := make([]byte, 64)
			if isInterMb && len(*prevY) == padW*padH {
				for r := 0; r < 16; r++ {
					for c := 0; c < 16; c++ {
						predY[r*16+c] = (*prevY)[(yY0+r)*padW+(yX0+c)]
					}
				}
				for r := 0; r < 8; r++ {
					for c := 0; c < 8; c++ {
						predU[r*8+c] = (*prevU)[(uvY0+r)*uvW+(uvX0+c)]
						predV[r*8+c] = (*prevV)[(uvY0+r)*uvW+(uvX0+c)]
					}
				}
			} else {
				var aboveY, leftY, aboveU, leftU, aboveV, leftV []byte
				if mbr > 0 {
					aboveY = reconY[(yY0-1)*padW+yX0 : (yY0-1)*padW+yX0+16]
					aboveU = reconU[(uvY0-1)*uvW+uvX0 : (uvY0-1)*uvW+uvX0+8]
					aboveV = reconV[(uvY0-1)*uvW+uvX0 : (uvY0-1)*uvW+uvX0+8]
				}
				if mbc > 0 {
					leftY = make([]byte, 16)
					for r := 0; r < 16; r++ {
						leftY[r] = reconY[(yY0+r)*padW+(yX0-1)]
					}
					leftU = make([]byte, 8)
					leftV = make([]byte, 8)
					for r := 0; r < 8; r++ {
						leftU[r] = reconU[(uvY0+r)*uvW+(uvX0-1)]
						leftV[r] = reconV[(uvY0+r)*uvW+(uvX0-1)]
					}
				}
				tlY := byte(129)
				if mbr > 0 && mbc > 0 {
					tlY = reconY[(yY0-1)*padW+(yX0-1)]
				} else if mbr > 0 {
					tlY = 127
				}
				tlU, tlV := byte(128), byte(128)
				if mbr > 0 && mbc > 0 {
					tlU = reconU[(uvY0-1)*uvW+(uvX0-1)]
					tlV = reconV[(uvY0-1)*uvW+(uvX0-1)]
				}
				Vp8PredictIntraMb(ymode, 16, aboveY, leftY, tlY, predY, 16)
				Vp8PredictIntraMb(uvMode, 8, aboveU, leftU, tlU, predU, 8)
				Vp8PredictIntraMb(uvMode, 8, aboveV, leftV, tlV, predV, 8)
			}

			var y2DcRec [16]int32
			var yAcDeq [16][16]int32
			var uDeq, vDeq [4][16]int32
			if !mbSkip {
				y2Coeffs, y2Nz := Vp8DecodeCoeffBlock(p1, 0, 0, qf.Y2Dc, qf.Y2Ac)
				y2DcRec = Vp8InverseWht4x4(&y2Coeffs)
				prevNz := 0
				if y2Nz {
					prevNz = 1
				}
				for sb := 0; sb < 16; sb++ {
					blk, nz := Vp8DecodeCoeffBlock(p1, 1, prevNz, qf.Y1Dc, qf.Y1Ac)
					yAcDeq[sb] = blk
					prevNz = 0
					if nz {
						prevNz = 1
					}
				}
				for sb := 0; sb < 4; sb++ {
					blk, nz := Vp8DecodeCoeffBlock(p1, 0, prevNz, qf.UvDc, qf.UvAc)
					uDeq[sb] = blk
					prevNz = 0
					if nz {
						prevNz = 1
					}
				}
				for sb := 0; sb < 4; sb++ {
					blk, nz := Vp8DecodeCoeffBlock(p1, 0, prevNz, qf.UvDc, qf.UvAc)
					vDeq[sb] = blk
					prevNz = 0
					if nz {
						prevNz = 1
					}
				}
			}
			for sbR := 0; sbR < 4; sbR++ {
				for sbC := 0; sbC < 4; sbC++ {
					sbIdx := sbR*4 + sbC
					deq := yAcDeq[sbIdx]
					deq[0] = y2DcRec[sbIdx]
					res := Vp8InverseIdct4x4(&deq)
					for r := 0; r < 4; r++ {
						for c := 0; c < 4; c++ {
							py := sbR*4 + r
							px := sbC*4 + c
							reconY[(yY0+py)*padW+(yX0+px)] = byte(clampI32(int32(predY[py*16+px])+int32(res[r*4+c]), 0, 255))
						}
					}
				}
			}
			for sbR := 0; sbR < 2; sbR++ {
				for sbC := 0; sbC < 2; sbC++ {
					sbIdx := sbR*2 + sbC
					resU := Vp8InverseIdct4x4(&uDeq[sbIdx])
					resV := Vp8InverseIdct4x4(&vDeq[sbIdx])
					for r := 0; r < 4; r++ {
						for c := 0; c < 4; c++ {
							py := sbR*4 + r
							px := sbC*4 + c
							reconU[(uvY0+py)*uvW+(uvX0+px)] = byte(clampI32(int32(predU[py*8+px])+int32(resU[r*4+c]), 0, 255))
							reconV[(uvY0+py)*uvW+(uvX0+px)] = byte(clampI32(int32(predV[py*8+px])+int32(resV[r*4+c]), 0, 255))
						}
					}
				}
			}
		}
	}
	if loopFilterLevel > 0 {
		applyVp8SimpleLoopFilter(reconY, padW, padH, loopFilterLevel)
	}
	cw, ch := w/2, h/2
	yCrop := make([]byte, w*h)
	uCrop := make([]byte, cw*ch)
	vCrop := make([]byte, cw*ch)
	for py := 0; py < h; py++ {
		copy(yCrop[py*w:(py+1)*w], reconY[py*padW:py*padW+w])
	}
	for cy := 0; cy < ch; cy++ {
		copy(uCrop[cy*cw:(cy+1)*cw], reconU[cy*uvW:cy*uvW+cw])
		copy(vCrop[cy*cw:(cy+1)*cw], reconV[cy*uvW:cy*uvW+cw])
	}
	i420Out := I420Surface{
		Width:  *curWidth,
		Height: *curHeight,
		YPlane: yCrop,
		UPlane: uCrop,
		VPlane: vCrop,
	}
	*prevY = reconY
	*prevU = reconU
	*prevV = reconV
	return VideoFrame{
		Width:    *curWidth,
		Height:   *curHeight,
		PtsMs:    ptsMs,
		Keyframe: header.IsKeyframe,
		Rgba:     i420Out.ToRgbaBt709(),
	}, nil
}

// EncodeVp8Rfc6386Bitstream encodes a sequence of VideoFrames as an RFC 6386 VP8 bitstream.
func EncodeVp8Rfc6386Bitstream(fps uint32, qp uint8, frames []VideoFrame) ([]byte, error) {
	if len(frames) == 0 {
		return nil, &VideoError{Kind: ErrInvalidHeader}
	}
	fps = maxU32(fps, 1)
	var prevY, prevU, prevV []byte
	firstPkt, err := EncodeVp8Rfc6386Frame(&frames[0], qp, true, &prevY, &prevU, &prevV)
	if err != nil {
		return nil, err
	}
	if len(frames) == 1 && fps == 30 {
		return firstPkt, nil
	}
	var out bytes.Buffer
	out.Write(firstPkt)
	out.WriteString("VP8F")
	var trailerHdr [8]byte
	binary.LittleEndian.PutUint32(trailerHdr[0:4], fps)
	binary.LittleEndian.PutUint32(trailerHdr[4:8], uint32(len(frames)-1))
	out.Write(trailerHdr[:])
	for idx := 1; idx < len(frames); idx++ {
		isKey := frames[idx].Keyframe || (idx%30 == 0)
		pkt, err := EncodeVp8Rfc6386Frame(&frames[idx], qp, isKey, &prevY, &prevU, &prevV)
		if err != nil {
			return nil, err
		}
		var pLen [4]byte
		binary.LittleEndian.PutUint32(pLen[:], uint32(len(pkt)))
		out.Write(pLen[:])
		out.Write(pkt)
	}
	return out.Bytes(), nil
}

// DecodeVp8Rfc6386Bitstream decodes an RFC 6386 VP8 bitstream into a VideoClip.
func DecodeVp8Rfc6386Bitstream(data []byte, hardwareDecode bool) (VideoClip, error) {
	var curW, curH uint32
	var prevY, prevU, prevV []byte
	firstFrame, err := DecodeVp8Rfc6386Frame(data, &curW, &curH, &prevY, &prevU, &prevV, 0)
	if err != nil {
		return VideoClip{}, err
	}
	frames := []VideoFrame{firstFrame}
	fps := uint32(30)
	if pos := bytes.Index(data, []byte("VP8F")); pos >= 0 && pos+12 <= len(data) {
		fps = maxU32(binary.LittleEndian.Uint32(data[pos+4:pos+8]), 1)
		extraCountU32 := binary.LittleEndian.Uint32(data[pos+8 : pos+12])
		if extraCountU32 >= MaxVideoFrames {
			return VideoClip{}, &VideoError{Kind: ErrFrameCountTooLarge, Count: extraCountU32 + 1}
		}
		extraCount := int(extraCountU32)
		cursor := pos + 12
		for idx := 0; idx < extraCount; idx++ {
			if cursor+4 > len(data) {
				return VideoClip{}, &VideoError{Kind: ErrCorruptedBitstream}
			}
			pktLen := int(binary.LittleEndian.Uint32(data[cursor : cursor+4]))
			cursor += 4
			if pktLen < 0 || pktLen > MaxPacketBytes || pktLen > len(data)-cursor {
				return VideoClip{}, &VideoError{Kind: ErrCorruptedBitstream}
			}
			ptsMs := uint32((uint64(idx+1) * 1000) / uint64(fps))
			f, err := DecodeVp8Rfc6386Frame(data[cursor:cursor+pktLen], &curW, &curH, &prevY, &prevU, &prevV, ptsMs)
			if err != nil {
				return VideoClip{}, err
			}
			frames = append(frames, f)
			cursor += pktLen
		}
	}
	return VideoClip{
		Width:   curW,
		Height:  curH,
		Fps:     fps,
		Backend: ProbePlatformGpuBackend(hardwareDecode),
		Profile: ProfileVp8Profile0,
		Frames:  frames,
	}, nil
}

// EncodeVp8Rfc6386Stream encodes VideoFrames as an RFC 6386 VP8 stream.
func EncodeVp8Rfc6386Stream(fps uint32, qp uint8, frames []VideoFrame) ([]byte, error) {
	return EncodeVp8Rfc6386Bitstream(fps, qp, frames)
}

// DecodeVp8Rfc6386Stream decodes an RFC 6386 VP8 stream (or legacy ADCT-wrapped VP8 stream).
func DecodeVp8Rfc6386Stream(data []byte, hardwareDecode bool) (VideoClip, error) {
	header, err := ParseVp8FrameHeader(data)
	if err != nil {
		return VideoClip{}, err
	}
	if !header.IsKeyframe || len(data) < 11 {
		return VideoClip{}, &VideoError{Kind: ErrInvalidHeader}
	}
	if err := ValidateVideoDimensions(uint32(header.Width), uint32(header.Height)); err != nil {
		return VideoClip{}, err
	}
	if len(data) >= 18 && string(data[14:18]) == "ADCT" {
		payloadLen := int(binary.BigEndian.Uint32(data[10:14]))
		if payloadLen < 0 || payloadLen > len(data)-14 {
			return VideoClip{}, &VideoError{Kind: ErrCorruptedBitstream}
		}
		return DecodeDctMcBitstream(data[14:14+payloadLen], hardwareDecode)
	}
	return DecodeVp8Rfc6386Bitstream(data, hardwareDecode)
}
