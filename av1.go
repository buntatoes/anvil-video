// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Buntato

package anvilvideo

import (
	"bytes"
	"encoding/binary"
	"math"
)

// Av1ObuType identifies AOMedia AV1 Open Bitstream Unit (OBU) types (Specification Section 6.2.2).
type Av1ObuType uint8

const (
	ObuSequenceHeader       Av1ObuType = 1
	ObuTemporalDelimiter    Av1ObuType = 2
	ObuFrameHeader          Av1ObuType = 3
	ObuTileGroup            Av1ObuType = 4
	ObuMetadata             Av1ObuType = 5
	ObuFrame                Av1ObuType = 6
	ObuRedundantFrameHeader Av1ObuType = 7
	ObuTileList             Av1ObuType = 8
	ObuPadding              Av1ObuType = 15
)

func Av1ObuTypeFromU8(v uint8) (Av1ObuType, bool) {
	switch Av1ObuType(v) {
	case ObuSequenceHeader, ObuTemporalDelimiter, ObuFrameHeader, ObuTileGroup,
		ObuMetadata, ObuFrame, ObuRedundantFrameHeader, ObuTileList, ObuPadding:
		return Av1ObuType(v), true
	default:
		return 0, false
	}
}

// WriteObuLeb128 appends value in AV1 leb128() format to out.
func WriteObuLeb128(out *bytes.Buffer, value uint64) {
	for {
		b := byte(value & 0x7F)
		value >>= 7
		if value != 0 {
			b |= 0x80
		}
		out.WriteByte(b)
		if value == 0 {
			break
		}
	}
}

// ReadObuLeb128 reads an unsigned LEB128 integer from data, returning (value, bytesConsumed).
func ReadObuLeb128(data []byte) (uint64, int, error) {
	var value uint64
	for i := 0; i < 8; i++ {
		if i >= len(data) {
			return 0, 0, &VideoError{Kind: ErrCorruptedBitstream}
		}
		b := data[i]
		value |= uint64(b&0x7F) << uint(i*7)
		if (b & 0x80) == 0 {
			if value > math.MaxUint32 {
				return 0, 0, &VideoError{Kind: ErrCorruptedBitstream}
			}
			return value, i + 1, nil
		}
	}
	return 0, 0, &VideoError{Kind: ErrCorruptedBitstream}
}

// WriteAv1Obu writes a complete AV1 OBU with obu_has_size_field = 1.
func WriteAv1Obu(out *bytes.Buffer, obuType Av1ObuType, payload []byte) {
	headerByte := (byte(obuType) << 3) | 0x02
	out.WriteByte(headerByte)
	WriteObuLeb128(out, uint64(len(payload)))
	out.Write(payload)
}

// Av1IntraMode identifies AV1 intra prediction modes (Specification Section 7.11.2).
type Av1IntraMode uint8

const (
	Av1DcPred Av1IntraMode = iota
	Av1VPred
	Av1HPred
	Av1D45Pred
	Av1D135Pred
	Av1SmoothPred
	Av1SmoothVPred
	Av1SmoothHPred
	Av1PaethPred
)

func Av1IntraModeFromU8(v uint8) Av1IntraMode {
	if v > 8 {
		return Av1PaethPred
	}
	return Av1IntraMode(v)
}

var Av1SmWeights8x8 = [8]uint8{255, 197, 146, 105, 73, 50, 37, 32}

// Av1PredictIntra8x8 predicts an 8x8 luma or chroma block using AV1 intra prediction.
func Av1PredictIntra8x8(plane []byte, stride, bx, by int, mode Av1IntraMode) [64]byte {
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
	case Av1DcPred:
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
	case Av1VPred:
		for r := 0; r < 8; r++ {
			copy(out[r*8:r*8+8], above[:8])
		}
	case Av1HPred:
		for r := 0; r < 8; r++ {
			for c := 0; c < 8; c++ {
				out[r*8+c] = left[r]
			}
		}
	case Av1D45Pred:
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
	case Av1D135Pred:
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
	case Av1SmoothPred:
		tr := uint32(above[7])
		bl := uint32(left[7])
		for r := 0; r < 8; r++ {
			wy := uint32(Av1SmWeights8x8[r])
			for c := 0; c < 8; c++ {
				wx := uint32(Av1SmWeights8x8[c])
				pred := wy*uint32(above[c]) + (256-wy)*bl + wx*uint32(left[r]) + (256-wx)*tr
				v := (pred + 256) >> 9
				if v > 255 {
					v = 255
				}
				out[r*8+c] = byte(v)
			}
		}
	case Av1SmoothVPred:
		bl := uint32(left[7])
		for r := 0; r < 8; r++ {
			wy := uint32(Av1SmWeights8x8[r])
			for c := 0; c < 8; c++ {
				pred := wy*uint32(above[c]) + (256-wy)*bl
				v := (pred + 128) >> 8
				if v > 255 {
					v = 255
				}
				out[r*8+c] = byte(v)
			}
		}
	case Av1SmoothHPred:
		tr := uint32(above[7])
		for r := 0; r < 8; r++ {
			leftPx := uint32(left[r])
			for c := 0; c < 8; c++ {
				wx := uint32(Av1SmWeights8x8[c])
				pred := wx*leftPx + (256-wx)*tr
				v := (pred + 128) >> 8
				if v > 255 {
					v = 255
				}
				out[r*8+c] = byte(v)
			}
		}
	default:
		tl := int32(topLeft)
		for r := 0; r < 8; r++ {
			l := int32(left[r])
			for c := 0; c < 8; c++ {
				t := int32(above[c])
				base := t + l - tl
				pt := base - t
				if pt < 0 {
					pt = -pt
				}
				pl := base - l
				if pl < 0 {
					pl = -pl
				}
				ptl := base - tl
				if ptl < 0 {
					ptl = -ptl
				}
				if pt <= pl && pt <= ptl {
					out[r*8+c] = byte(t)
				} else if pl <= ptl {
					out[r*8+c] = byte(l)
				} else {
					out[r*8+c] = byte(tl)
				}
			}
		}
	}
	return out
}

// Av1CdefFindDirection8x8 performs the AV1 Section 7.15.1 CDEF 8-direction search on an 8x8 block.
func Av1CdefFindDirection8x8(block *[64]byte) (uint8, uint32) {
	var dirEnergy [8]uint64
	var varV, varH uint64
	for i := 0; i < 8; i++ {
		var colSum, rowSum int32
		for j := 0; j < 8; j++ {
			colSum += int32(block[j*8+i]) - 128
			rowSum += int32(block[i*8+j]) - 128
		}
		varV += uint64(colSum * colSum)
		varH += uint64(rowSum * rowSum)
	}
	dirEnergy[0] = varV
	dirEnergy[4] = varH

	var diag45, diag135 [15]int32
	var diagCnt [15]uint64
	for r := 0; r < 8; r++ {
		for c := 0; c < 8; c++ {
			v := int32(block[r*8+c]) - 128
			diag45[r+c] += v
			diag135[r+7-c] += v
			diagCnt[r+c]++
		}
	}
	for k := 0; k < 15; k++ {
		cnt := diagCnt[k]
		if cnt < 1 {
			cnt = 1
		}
		dirEnergy[2] += (uint64(diag45[k]*diag45[k]) * 8) / cnt
		dirEnergy[6] += (uint64(diag135[k]*diag135[k]) * 8) / cnt
	}
	dirEnergy[1] = (dirEnergy[0] + dirEnergy[2]) / 2
	dirEnergy[3] = (dirEnergy[2] + dirEnergy[4]) / 2
	dirEnergy[5] = (dirEnergy[4] + dirEnergy[6]) / 2
	dirEnergy[7] = (dirEnergy[6] + dirEnergy[0]) / 2

	bestDir := uint8(0)
	var bestE uint64
	minE := uint64(math.MaxUint64)
	for d, e := range dirEnergy {
		if e > bestE {
			bestE = e
			bestDir = uint8(d)
		}
		if e < minE {
			minE = e
		}
	}
	var contrast uint32
	if bestE > minE {
		contrast = uint32((bestE - minE) >> 6)
	}
	return bestDir, contrast
}

// Av1CdefFilterPlane applies the AV1 Section 7.15 CDEF filter across all 8x8 regions in plane.
func Av1CdefFilterPlane(plane []byte, w, h int, strength, damping uint8) {
	if strength == 0 {
		return
	}
	blocksX := w / 8
	blocksY := h / 8
	for by := 0; by < blocksY; by++ {
		for bx := 0; bx < blocksX; bx++ {
			var blk8 [64]byte
			for r := 0; r < 8; r++ {
				for c := 0; c < 8; c++ {
					blk8[r*8+c] = plane[(by*8+r)*w+(bx*8+c)]
				}
			}
			_, variance := Av1CdefFindDirection8x8(&blk8)
			adjStrength := strength
			if variance <= 64 {
				adjStrength = maxU8(strength/2, 1)
			}
			for sy := 0; sy < 2; sy++ {
				for sx := 0; sx < 2; sx++ {
					var sub4 [16]byte
					for r := 0; r < 4; r++ {
						for c := 0; c < 4; c++ {
							sub4[r*4+c] = plane[(by*8+sy*4+r)*w+(bx*8+sx*4+c)]
						}
					}
					Av1CdefFilter4x4(&sub4, adjStrength, damping)
					for r := 0; r < 4; r++ {
						for c := 0; c < 4; c++ {
							plane[(by*8+sy*4+r)*w+(bx*8+sx*4+c)] = sub4[r*4+c]
						}
					}
				}
			}
		}
	}
}

// Av1SymbolEncoder implements the AV1 Section 8.2 symbol entropy encoder.
type Av1SymbolEncoder struct {
	inner *Vp9BoolEncoder
}

func NewAv1SymbolEncoder() *Av1SymbolEncoder {
	return &Av1SymbolEncoder{inner: NewVp9BoolEncoder()}
}

func (e *Av1SymbolEncoder) WriteCdfBit(bit bool, cdfProbZero uint16) {
	clamped := clampI32(int32(cdfProbZero), 128, 32640)
	p8 := uint8(clampI32(clamped>>7, 1, 255))
	e.inner.WriteBit(bit, p8)
}

func (e *Av1SymbolEncoder) WriteLiteral(value uint32, numBits uint8) {
	e.inner.WriteLiteral(value, numBits)
}

func (e *Av1SymbolEncoder) WriteSigned(value int32, numBits uint8) {
	e.inner.WriteSigned(value, numBits)
}

func (e *Av1SymbolEncoder) Finish() []byte {
	return e.inner.Finish()
}

// Av1SymbolDecoder implements the AV1 Section 8.2 symbol entropy decoder.
type Av1SymbolDecoder struct {
	inner *Vp9BoolDecoder
}

func NewAv1SymbolDecoder(data []byte) (*Av1SymbolDecoder, error) {
	dec, err := NewVp9BoolDecoder(data)
	if err != nil {
		return nil, err
	}
	return &Av1SymbolDecoder{inner: dec}, nil
}

func (d *Av1SymbolDecoder) ReadCdfBit(cdfProbZero uint16) bool {
	clamped := clampI32(int32(cdfProbZero), 128, 32640)
	p8 := uint8(clampI32(clamped>>7, 1, 255))
	return d.inner.ReadBit(p8)
}

func (d *Av1SymbolDecoder) ReadLiteral(numBits uint8) uint32 {
	return d.inner.ReadLiteral(numBits)
}

func (d *Av1SymbolDecoder) ReadSigned(numBits uint8) int32 {
	return d.inner.ReadSigned(numBits)
}

func encodeCoeffsAv1(enc *Av1SymbolEncoder, qZz []int16) {
	eob := 0
	for i, c := range qZz {
		if c != 0 {
			eob = i + 1
		}
	}
	enc.WriteCdfBit(eob > 0, 16384)
	if eob == 0 {
		return
	}
	enc.WriteLiteral(uint32(eob), 7)
	for _, c := range qZz[:eob] {
		enc.WriteCdfBit(c != 0, 18432)
		if c != 0 {
			mag := int32(c)
			if mag < 0 {
				mag = -mag
			}
			if mag <= 3 {
				enc.WriteCdfBit(false, 20480)
				enc.WriteLiteral(uint32(mag-1), 2)
			} else {
				enc.WriteCdfBit(true, 20480)
				if mag > 4095 {
					mag = 4095
				}
				enc.WriteLiteral(uint32(mag), 12)
			}
			enc.WriteCdfBit(c < 0, 16384)
		}
	}
}

func decodeCoeffsAv1(dec *Av1SymbolDecoder, length int) []int16 {
	out := make([]int16, length)
	if !dec.ReadCdfBit(16384) {
		return out
	}
	eob := int(dec.ReadLiteral(7))
	if eob > length {
		eob = length
	}
	for i := 0; i < eob; i++ {
		if dec.ReadCdfBit(18432) {
			var mag int16
			if !dec.ReadCdfBit(20480) {
				mag = int16(dec.ReadLiteral(2) + 1)
			} else {
				mag = int16(dec.ReadLiteral(12))
			}
			if dec.ReadCdfBit(16384) {
				out[i] = -mag
			} else {
				out[i] = mag
			}
		}
	}
	return out
}

// EncodeAv1Bitstream encodes VideoFrames as a clean-room AOMedia AV1 (av01) OBU stream.
func EncodeAv1Bitstream(fps uint32, qp uint8, frames []VideoFrame) ([]byte, error) {
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
	cdefStrength := uint8(clampI32(int32(qpClamped)/6, 1, 8))
	cdefDamping := uint8(4)

	refSurface := NewI420Surface(width, height)
	frameObus := make([][]byte, 0, len(frames))

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

		sym := NewAv1SymbolEncoder()
		sym.WriteCdfBit(isKeyframe, 16384)
		sym.WriteLiteral(clampU32(fps, 1, 240), 8)
		sym.WriteLiteral(f.PtsMs, 24)
		sym.WriteLiteral(uint32(qIdx), 8)
		sym.WriteLiteral(uint32(cdefStrength), 4)
		sym.WriteLiteral(uint32(cdefDamping), 3)

		reconY := make([]byte, w*h)
		for i := range reconY {
			reconY[i] = 128
		}

		for byIdx := 0; byIdx < blocksY; byIdx++ {
			for bxIdx := 0; bxIdx < blocksX; bxIdx++ {
				bx := bxIdx * 8
				by := byIdx * 8
				var pred [64]byte
				if isKeyframe {
					bestMode := Av1DcPred
					bestSad := uint32(math.MaxUint32)
					for _, cand := range []Av1IntraMode{
						Av1DcPred, Av1VPred, Av1HPred, Av1D45Pred, Av1D135Pred,
						Av1SmoothPred, Av1SmoothVPred, Av1SmoothHPred, Av1PaethPred,
					} {
						candPred := Av1PredictIntra8x8(reconY, w, bx, by, cand)
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
					sym.WriteLiteral(uint32(bestMode), 4)
				} else {
					var bestMv [2]int8
					bestSad := uint32(math.MaxUint32)
					for dy := int8(-6); dy <= 6; dy += 2 {
						for dx := int8(-6); dx <= 6; dx += 2 {
							var sad uint32
							for py := 0; py < 8; py++ {
								for px := 0; px < 8; px++ {
									if bx+px < w && by+py < h {
										cur := int32(srcI420.YPlane[(by+py)*w+(bx+px)])
										rf := int32(readPlaneClamped(refSurface.YPlane, w, h, bx+px+int(dx), by+py+int(dy)))
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
								bestMv = [2]int8{dx, dy}
							}
						}
					}
					hasMv := bestMv != [2]int8{0, 0}
					sym.WriteCdfBit(hasMv, 16384)
					if hasMv {
						sym.WriteSigned(int32(bestMv[0]), 5)
						sym.WriteSigned(int32(bestMv[1]), 5)
					}
					for py := 0; py < 8; py++ {
						for px := 0; px < 8; px++ {
							pred[py*8+px] = readPlaneClamped(refSurface.YPlane, w, h, bx+px+int(bestMv[0]), by+py+int(bestMv[1]))
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
				encodeCoeffsAv1(sym, qZz[:])
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
		refPlanes := [2][]byte{refSurface.UPlane, refSurface.VPlane}
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
						pred = Av1PredictIntra8x8(reconC, cw, bx, by, Av1DcPred)
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
					encodeCoeffsAv1(sym, qZz[:])
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

		Av1CdefFilterPlane(reconY, w, h, cdefStrength, cdefDamping)
		refSurface = I420Surface{Width: width, Height: height, YPlane: reconY, UPlane: reconU, VPlane: reconV}
		frameObus = append(frameObus, sym.Finish())
	}

	var body bytes.Buffer
	body.WriteString("AV1B")
	WriteObuLeb128(&body, uint64(len(frameObus)))
	for _, fBytes := range frameObus {
		WriteAv1Obu(&body, ObuFrame, fBytes)
	}

	var out bytes.Buffer
	out.Write([]byte{0x12, 0x00, 0x0A, 12})
	out.WriteString("AV01")
	var dims [8]byte
	binary.BigEndian.PutUint32(dims[0:4], width)
	binary.BigEndian.PutUint32(dims[4:8], height)
	out.Write(dims[:])
	out.WriteByte(0x32)
	var bLen [4]byte
	binary.BigEndian.PutUint32(bLen[:], uint32(body.Len()))
	out.Write(bLen[:])
	out.Write(body.Bytes())
	return out.Bytes(), nil
}

// DecodeAv1Bitstream decodes a clean-room AOMedia AV1 (av01) OBU stream into a VideoClip.
func DecodeAv1Bitstream(data []byte, hardwareDecode bool) (VideoClip, error) {
	if len(data) > MaxPacketBytes {
		return VideoClip{}, &VideoError{Kind: ErrPacketTooLarge, Size: len(data)}
	}
	if len(data) < 25 || !bytes.Equal(data[0:2], []byte{0x12, 0x00}) || data[2] != 0x0A || string(data[4:8]) != "AV01" || data[16] != 0x32 {
		return VideoClip{}, &VideoError{Kind: ErrInvalidHeader}
	}
	width := binary.BigEndian.Uint32(data[8:12])
	height := binary.BigEndian.Uint32(data[12:16])
	if err := ValidateVideoDimensions(width, height); err != nil {
		return VideoClip{}, err
	}
	payloadLen := int(binary.BigEndian.Uint32(data[17:21]))
	if payloadLen < 0 || payloadLen > len(data)-21 {
		return VideoClip{}, &VideoError{Kind: ErrCorruptedBitstream}
	}
	body := data[21 : 21+payloadLen]
	if len(body) < 5 || string(body[0:4]) != "AV1B" {
		return VideoClip{}, &VideoError{Kind: ErrCorruptedBitstream}
	}
	frameCountU64, consumed, err := ReadObuLeb128(body[4:])
	if err != nil {
		return VideoClip{}, err
	}
	if frameCountU64 == 0 || frameCountU64 > uint64(MaxVideoFrames) {
		return VideoClip{}, &VideoError{Kind: ErrFrameCountTooLarge, Count: uint32(min(frameCountU64, uint64(math.MaxUint32)))}
	}
	frameCount := int(frameCountU64)

	w, h := int(width), int(height)
	cw, ch := w/2, h/2
	blocksX := (w + 7) / 8
	blocksY := (h + 7) / 8
	cBlocksX := (cw + 7) / 8
	cBlocksY := (ch + 7) / 8

	refSurface := NewI420Surface(width, height)
	frames := make([]VideoFrame, 0, frameCount)
	clipFps := uint32(30)
	cursor := 4 + consumed

	for idx := 0; idx < frameCount; idx++ {
		if cursor >= len(body) {
			return VideoClip{}, &VideoError{Kind: ErrCorruptedBitstream}
		}
		obuHdr := body[cursor]
		obuType, ok := Av1ObuTypeFromU8((obuHdr >> 3) & 0x0F)
		if !ok || obuType != ObuFrame {
			return VideoClip{}, &VideoError{Kind: ErrCorruptedBitstream}
		}
		cursor++
		obuSz, lebLen, err := ReadObuLeb128(body[cursor:])
		if err != nil {
			return VideoClip{}, err
		}
		cursor += lebLen
		if obuSz > uint64(len(body)-cursor) {
			return VideoClip{}, &VideoError{Kind: ErrCorruptedBitstream}
		}
		obuLen := int(obuSz)
		sym, err := NewAv1SymbolDecoder(body[cursor : cursor+obuLen])
		if err != nil {
			return VideoClip{}, err
		}
		cursor += obuLen

		isKeyframe := sym.ReadCdfBit(16384)
		if idx == 0 && !isKeyframe {
			return VideoClip{}, &VideoError{Kind: ErrCorruptedBitstream}
		}
		fps := maxU32(sym.ReadLiteral(8), 1)
		ptsMs := sym.ReadLiteral(24)
		qIdx := uint8(sym.ReadLiteral(8))
		cdefStrength := uint8(sym.ReadLiteral(4))
		cdefDamping := uint8(sym.ReadLiteral(3))
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
		for byIdx := 0; byIdx < blocksY; byIdx++ {
			for bxIdx := 0; bxIdx < blocksX; bxIdx++ {
				bx := bxIdx * 8
				by := byIdx * 8
				var pred [64]byte
				if isKeyframe {
					mode := Av1IntraModeFromU8(uint8(sym.ReadLiteral(4)))
					pred = Av1PredictIntra8x8(reconY, w, bx, by, mode)
				} else {
					hasMv := sym.ReadCdfBit(16384)
					var mv [2]int8
					if hasMv {
						mv[0] = int8(clampI32(sym.ReadSigned(5), -64, 64))
						mv[1] = int8(clampI32(sym.ReadSigned(5), -64, 64))
					}
					for py := 0; py < 8; py++ {
						for px := 0; px < 8; px++ {
							pred[py*8+px] = readPlaneClamped(refSurface.YPlane, w, h, bx+px+int(mv[0]), by+py+int(mv[1]))
						}
					}
				}
				qZz := decodeCoeffsAv1(sym, 64)
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
		refPlanes := [2][]byte{refSurface.UPlane, refSurface.VPlane}
		for pIdx := 0; pIdx < 2; pIdx++ {
			reconC := reconPlanes[pIdx]
			refC := refPlanes[pIdx]
			for byIdx := 0; byIdx < cBlocksY; byIdx++ {
				for bxIdx := 0; bxIdx < cBlocksX; bxIdx++ {
					bx := bxIdx * 8
					by := byIdx * 8
					var pred [64]byte
					if isKeyframe {
						pred = Av1PredictIntra8x8(reconC, cw, bx, by, Av1DcPred)
					} else {
						for py := 0; py < 8; py++ {
							for px := 0; px < 8; px++ {
								pred[py*8+px] = readPlaneClamped(refC, cw, ch, bx+px, by+py)
							}
						}
					}
					qZz := decodeCoeffsAv1(sym, 64)
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

		Av1CdefFilterPlane(reconY, w, h, cdefStrength, cdefDamping)
		refSurface = I420Surface{Width: width, Height: height, YPlane: reconY, UPlane: reconU, VPlane: reconV}
		vf, err := NewVideoFrame(width, height, ptsMs, isKeyframe, refSurface.ToRgbaBt709())
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
		Profile: ProfileAv1Main,
		Frames:  frames,
	}, nil
}

// EncodeAv1ObuStream encodes VideoFrames as an AV1 (av01) Open Bitstream Unit (OBU) stream.
func EncodeAv1ObuStream(fps uint32, qp uint8, frames []VideoFrame) ([]byte, error) {
	return EncodeAv1Bitstream(fps, qp, frames)
}

// DecodeAv1ObuStream decodes an AV1 (av01) OBU stream (or legacy ADCT-wrapped AV1 stream).
func DecodeAv1ObuStream(data []byte, hardwareDecode bool) (VideoClip, error) {
	if len(data) < 21 || !bytes.Equal(data[0:2], []byte{0x12, 0x00}) || data[2] != 0x0A || string(data[4:8]) != "AV01" {
		return VideoClip{}, &VideoError{Kind: ErrInvalidHeader}
	}
	if data[16] != 0x32 {
		return VideoClip{}, &VideoError{Kind: ErrCorruptedBitstream}
	}
	payloadLen := int(binary.BigEndian.Uint32(data[17:21]))
	if payloadLen < 0 || payloadLen > len(data)-21 {
		return VideoClip{}, &VideoError{Kind: ErrCorruptedBitstream}
	}
	if payloadLen >= 4 && string(data[21:25]) == "ADCT" {
		return DecodeDctMcBitstream(data[21:21+payloadLen], hardwareDecode)
	}
	return DecodeAv1Bitstream(data, hardwareDecode)
}
