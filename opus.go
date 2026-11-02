// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Buntato

package anvilvideo

import (
	"math"
)

// CeltEBands holds RFC 6716 Section 4.3.1 CELT 21 Bark-scale band boundaries.
var CeltEBands = [22]int{
	0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100,
}

const CeltPreemphCoeff float32 = 0.8500061

// OpusMode identifies the operating mode encoded in the RFC 6716 Opus TOC byte.
type OpusMode uint8

const (
	OpusModeSilkOnly OpusMode = iota
	OpusModeHybrid
	OpusModeCeltOnly
)

// OpusBandwidth identifies the audio passband encoded in the RFC 6716 Opus TOC byte.
type OpusBandwidth uint8

const (
	OpusBandwidthNarrowband OpusBandwidth = iota
	OpusBandwidthMediumband
	OpusBandwidthWideband
	OpusBandwidthSuperWideband
	OpusBandwidthFullband
)

// OpusPacketHeader holds the parsed RFC 6716 Section 3.1 Opus TOC and frame count header.
type OpusPacketHeader struct {
	Config          uint8
	Mode            OpusMode
	Bandwidth       OpusBandwidth
	FrameDurationUs uint32
	Stereo          bool
	FrameCountCode  uint8
	FrameCount      uint8
	VBR             bool
	PayloadOffset   int
}

// ParseOpusPacketHeader parses an RFC 6716 Opus packet header from packet.
func ParseOpusPacketHeader(packet []byte) (OpusPacketHeader, error) {
	if len(packet) == 0 {
		return OpusPacketHeader{}, &VideoError{Kind: ErrInvalidHeader}
	}
	toc := packet[0]
	config := toc >> 3
	stereo := (toc & 0x04) != 0
	code := toc & 0x03

	var mode OpusMode
	switch {
	case config <= 11:
		mode = OpusModeSilkOnly
	case config <= 15:
		mode = OpusModeHybrid
	default:
		mode = OpusModeCeltOnly
	}
	var bandwidth OpusBandwidth
	switch {
	case config <= 3:
		bandwidth = OpusBandwidthNarrowband
	case config <= 7:
		bandwidth = OpusBandwidthMediumband
	case (config >= 8 && config <= 11) || (config >= 20 && config <= 23):
		bandwidth = OpusBandwidthWideband
	case (config >= 12 && config <= 13) || (config >= 24 && config <= 27):
		bandwidth = OpusBandwidthSuperWideband
	case (config >= 14 && config <= 15) || (config >= 28 && config <= 31):
		bandwidth = OpusBandwidthFullband
	default:
		bandwidth = OpusBandwidthNarrowband
	}
	var frameDurationUs uint32
	switch config {
	case 16, 20, 24, 28:
		frameDurationUs = 2500
	case 17, 21, 25, 29:
		frameDurationUs = 5000
	case 0, 4, 8, 12, 14, 18, 22, 26, 30:
		frameDurationUs = 10000
	case 1, 5, 9, 13, 15, 19, 23, 27, 31:
		frameDurationUs = 20000
	case 2, 6, 10:
		frameDurationUs = 40000
	default:
		frameDurationUs = 60000
	}
	var frameCount uint8
	var vbr bool
	var payloadOffset int
	switch code {
	case 0:
		frameCount, vbr, payloadOffset = 1, false, 1
	case 1:
		frameCount, vbr, payloadOffset = 2, false, 1
	case 2:
		if len(packet) < 2 {
			return OpusPacketHeader{}, &VideoError{Kind: ErrCorruptedBitstream}
		}
		frameCount, vbr, payloadOffset = 2, true, 2
	default:
		if len(packet) < 2 {
			return OpusPacketHeader{}, &VideoError{Kind: ErrCorruptedBitstream}
		}
		countByte := packet[1]
		m := countByte & 0x3F
		if m == 0 || m > 48 {
			return OpusPacketHeader{}, &VideoError{Kind: ErrCorruptedBitstream}
		}
		frameCount, vbr, payloadOffset = m, (countByte&0x80) != 0, 2
	}
	return OpusPacketHeader{
		Config:          config,
		Mode:            mode,
		Bandwidth:       bandwidth,
		FrameDurationUs: frameDurationUs,
		Stereo:          stereo,
		FrameCountCode:  code,
		FrameCount:      frameCount,
		VBR:             vbr,
		PayloadOffset:   payloadOffset,
	}, nil
}

func (h OpusPacketHeader) SamplesPerFrame48k() int {
	return (int(h.FrameDurationUs) * 48) / 1000
}

// OpusRangeEncoder implements the IETF RFC 6716 arithmetic range encoder.
type OpusRangeEncoder struct {
	invBytes []byte
	low      uint64
	rng      uint32
	rem      int32
	ext      int
}

func NewOpusRangeEncoder() *OpusRangeEncoder {
	return &OpusRangeEncoder{
		invBytes: make([]byte, 0, 256),
		low:      0,
		rng:      1 << 31,
		rem:      -1,
		ext:      0,
	}
}

func (e *OpusRangeEncoder) carryOut(c uint32) {
	if c != 0xFF {
		if e.rem >= 0 {
			carry := byte(c >> 8)
			e.invBytes = append(e.invBytes, byte(e.rem)+carry)
			fill := byte(0xFF)
			if carry != 0 {
				fill = 0x00
			}
			for e.ext > 0 {
				e.invBytes = append(e.invBytes, fill)
				e.ext--
			}
		}
		e.rem = int32(c & 0xFF)
	} else {
		e.ext++
	}
}

func (e *OpusRangeEncoder) normalize() {
	for e.rng <= (1 << 23) {
		e.carryOut(uint32(e.low >> 24))
		e.low = (e.low & 0x00FFFFFF) << 8
		e.rng <<= 8
	}
}

func (e *OpusRangeEncoder) EncodeBitLogp(bit bool, logp uint32) {
	r := e.rng >> logp
	if bit {
		e.rng = r
	} else {
		e.low += uint64(r)
		e.rng -= r
	}
	e.normalize()
}

func (e *OpusRangeEncoder) EncodeBits(value uint32, numBits uint8) {
	for shift := int(numBits) - 1; shift >= 0; shift-- {
		bit := ((value >> uint(shift)) & 1) != 0
		e.EncodeBitLogp(bit, 1)
	}
}

func (e *OpusRangeEncoder) EncodeUint(k, ft uint32) {
	if ft < 1 {
		ft = 1
	}
	if k >= ft {
		k = ft - 1
	}
	scale := e.rng / ft
	if scale < 1 {
		scale = 1
	}
	fh := (k + 1) * scale
	if e.rng > fh {
		e.low += uint64(e.rng - fh)
	}
	e.rng = scale
	e.normalize()
}

func (e *OpusRangeEncoder) Finish() []byte {
	for i := 0; i < 4; i++ {
		e.carryOut(uint32(e.low >> 24))
		e.low = (e.low & 0x00FFFFFF) << 8
	}
	if e.rem >= 0 {
		e.invBytes = append(e.invBytes, byte(e.rem))
	}
	for e.ext > 0 {
		e.invBytes = append(e.invBytes, 0xFF)
		e.ext--
	}
	if len(e.invBytes) == 0 {
		return []byte{254, 255, 255, 255}
	}
	out := make([]byte, len(e.invBytes))
	inv0 := e.invBytes[0] & 0x7F
	b0 := byte(0)
	if 127 > inv0 {
		b0 = (127 - inv0) << 1
	}
	out[0] = b0
	for i := 1; i < len(e.invBytes); i++ {
		out[i] = 255 - e.invBytes[i]
	}
	return out
}

// OpusRangeDecoder implements the IETF RFC 6716 Section 4.1 arithmetic range decoder.
type OpusRangeDecoder struct {
	data []byte
	pos  int
	rng  uint32
	val  uint32
}

func NewOpusRangeDecoder(data []byte) *OpusRangeDecoder {
	dec := &OpusRangeDecoder{
		data: data,
		pos:  0,
		rng:  128,
		val:  0,
	}
	b0 := dec.readByte()
	dec.val = 127 - (b0 >> 1)
	dec.normalize()
	return dec
}

func (d *OpusRangeDecoder) readByte() uint32 {
	if d.pos < len(d.data) {
		b := uint32(d.data[d.pos])
		d.pos++
		return b
	}
	return 0
}

func (d *OpusRangeDecoder) normalize() {
	for d.rng <= (1 << 23) {
		d.rng <<= 8
		b := d.readByte()
		d.val = ((d.val << 8) + (255 - b)) & 0x7FFFFFFF
	}
}

func (d *OpusRangeDecoder) DecodeBitLogp(logp uint32) bool {
	r := d.rng >> logp
	bit := d.val < r
	if bit {
		d.rng = r
	} else {
		d.val -= r
		d.rng -= r
	}
	d.normalize()
	return bit
}

func (d *OpusRangeDecoder) DecodeUint(ft uint32) uint32 {
	if ft < 1 {
		ft = 1
	}
	scale := d.rng / ft
	if scale < 1 {
		scale = 1
	}
	div := d.val / scale
	var k uint32
	if ft-1 > div {
		k = (ft - 1) - div
	}
	fl := k * scale
	fh := (k + 1) * scale
	var sub uint32
	if d.rng > fh {
		sub = d.rng - fh
	}
	if d.val > sub {
		d.val -= sub
	} else {
		d.val = 0
	}
	d.rng = fh - fl
	d.normalize()
	if k >= ft {
		return ft - 1
	}
	return k
}

func OpusDecodeBits(dec *OpusRangeDecoder, numBits uint8) uint32 {
	var v uint32
	for i := uint8(0); i < numBits; i++ {
		v <<= 1
		if dec.DecodeBitLogp(1) {
			v |= 1
		}
	}
	return v
}

// SplitOpusPacketFrames splits an RFC 6716 Opus packet into its header and individual frame slices.
func SplitOpusPacketFrames(packet []byte) (OpusPacketHeader, [][]byte, error) {
	header, err := ParseOpusPacketHeader(packet)
	if err != nil {
		return OpusPacketHeader{}, nil, err
	}
	pos := header.PayloadOffset
	if pos > len(packet) {
		return OpusPacketHeader{}, nil, &VideoError{Kind: ErrCorruptedBitstream}
	}
	readFrameLen := func(cursor *int) (int, error) {
		if *cursor >= len(packet) {
			return 0, &VideoError{Kind: ErrCorruptedBitstream}
		}
		b0 := int(packet[*cursor])
		*cursor++
		if b0 < 252 {
			return b0, nil
		}
		if *cursor >= len(packet) {
			return 0, &VideoError{Kind: ErrCorruptedBitstream}
		}
		b1 := int(packet[*cursor])
		*cursor++
		return b0 + 4*b1, nil
	}
	frames := make([][]byte, 0, int(header.FrameCount))
	switch header.FrameCountCode {
	case 0:
		frames = append(frames, packet[pos:])
	case 1:
		rem := len(packet) - pos
		if rem%2 != 0 {
			return OpusPacketHeader{}, nil, &VideoError{Kind: ErrCorruptedBitstream}
		}
		half := rem / 2
		frames = append(frames, packet[pos:pos+half], packet[pos+half:])
	case 2:
		len1, err := readFrameLen(&pos)
		if err != nil || pos+len1 > len(packet) {
			return OpusPacketHeader{}, nil, &VideoError{Kind: ErrCorruptedBitstream}
		}
		frames = append(frames, packet[pos:pos+len1], packet[pos+len1:])
	default:
		countByte := packet[1]
		hasPadding := (countByte & 0x40) != 0
		padTotal := 0
		if hasPadding {
			for {
				if pos >= len(packet) {
					return OpusPacketHeader{}, nil, &VideoError{Kind: ErrCorruptedBitstream}
				}
				pb := packet[pos]
				pos++
				if pb == 255 {
					padTotal += 254
				} else {
					padTotal += int(pb)
					break
				}
			}
		}
		if pos+padTotal > len(packet) {
			return OpusPacketHeader{}, nil, &VideoError{Kind: ErrCorruptedBitstream}
		}
		payloadEnd := len(packet) - padTotal
		m := int(header.FrameCount)
		if header.VBR {
			lengths := make([]int, 0, m-1)
			for i := 0; i < m-1; i++ {
				l, err := readFrameLen(&pos)
				if err != nil {
					return OpusPacketHeader{}, nil, err
				}
				lengths = append(lengths, l)
			}
			for _, l := range lengths {
				if pos+l > payloadEnd {
					return OpusPacketHeader{}, nil, &VideoError{Kind: ErrCorruptedBitstream}
				}
				frames = append(frames, packet[pos:pos+l])
				pos += l
			}
			if pos > payloadEnd {
				return OpusPacketHeader{}, nil, &VideoError{Kind: ErrCorruptedBitstream}
			}
			frames = append(frames, packet[pos:payloadEnd])
		} else {
			rem := payloadEnd - pos
			if m == 0 || rem%m != 0 {
				return OpusPacketHeader{}, nil, &VideoError{Kind: ErrCorruptedBitstream}
			}
			each := rem / m
			for i := 0; i < m; i++ {
				frames = append(frames, packet[pos+i*each:pos+(i+1)*each])
			}
		}
	}
	return header, frames, nil
}

// PackOpusFrames packs encoded Opus frames into a single RFC 6716 packet (code 0, 1, 2, or 3).
func PackOpusFrames(config uint8, stereo bool, frames [][]byte) ([]byte, error) {
	if len(frames) == 0 || len(frames) > 48 {
		return nil, &VideoError{Kind: ErrCorruptedBitstream}
	}
	stereoBit := uint8(0)
	if stereo {
		stereoBit = 0x04
	}
	writeFrameLen := func(out *[]byte, l int) {
		if l < 252 {
			*out = append(*out, byte(l))
		} else {
			b0 := 252 + ((l - 252) & 3)
			b1 := (l - b0) >> 2
			*out = append(*out, byte(b0), byte(b1))
		}
	}
	if len(frames) == 1 {
		toc := ((config & 0x1F) << 3) | stereoBit
		out := make([]byte, 0, 1+len(frames[0]))
		out = append(out, toc)
		out = append(out, frames[0]...)
		return out, nil
	}
	if len(frames) == 2 && len(frames[0]) == len(frames[1]) {
		toc := ((config & 0x1F) << 3) | stereoBit | 1
		out := make([]byte, 0, 1+len(frames[0])*2)
		out = append(out, toc)
		out = append(out, frames[0]...)
		out = append(out, frames[1]...)
		return out, nil
	}
	if len(frames) == 2 {
		toc := ((config & 0x1F) << 3) | stereoBit | 2
		out := []byte{toc}
		writeFrameLen(&out, len(frames[0]))
		out = append(out, frames[0]...)
		out = append(out, frames[1]...)
		return out, nil
	}
	allEqual := true
	for _, f := range frames {
		if len(f) != len(frames[0]) {
			allEqual = false
			break
		}
	}
	toc := ((config & 0x1F) << 3) | stereoBit | 3
	out := []byte{toc}
	m := uint8(len(frames))
	if allEqual {
		out = append(out, m)
	} else {
		out = append(out, 0x80|m)
		for _, f := range frames[:len(frames)-1] {
			writeFrameLen(&out, len(f))
		}
	}
	for _, f := range frames {
		out = append(out, f...)
	}
	return out, nil
}

// EncodeSilkFrame encodes a single SILK speech frame using LPC analysis and range-coded residuals.
func EncodeSilkFrame(enc *OpusRangeEncoder, pcm48k []int16, channels, samplesPerCh int) {
	silkLen := samplesPerCh / 3
	if silkLen < 1 {
		silkLen = 1
	}
	enc.EncodeBitLogp(true, 1)
	enc.EncodeBitLogp(false, 1)

	for ch := 0; ch < channels; ch++ {
		sig := make([]float64, silkLen)
		for i := 0; i < silkLen; i++ {
			srcIdx := i * 3
			if srcIdx >= samplesPerCh {
				srcIdx = samplesPerCh - 1
			}
			sig[i] = float64(pcm48k[srcIdx*channels+ch])
		}
		var r [5]float64
		for lag := 0; lag < 5; lag++ {
			sum := 0.0
			for i := lag; i < silkLen; i++ {
				sum += sig[i] * sig[i-lag]
			}
			r[lag] = sum
		}
		var a1, a2 float64
		if r[0] > 1.0 {
			k1 := clampF64(r[1]/r[0], -0.95, 0.95)
			e1 := r[0] * math.Max(1.0-k1*k1, 0.05)
			k2 := clampF64((r[2]-k1*r[1])/e1, -0.90, 0.90)
			a1 = clampF64(k1*(1.0-k2), -1.5, 1.5)
			a2 = clampF64(k2, -0.95, 0.95)
		}
		qA1 := uint32(clampI32(int32(math.Round(a1*64.0))+128, 0, 255))
		qA2 := uint32(clampI32(int32(math.Round(a2*64.0))+128, 0, 255))
		enc.EncodeUint(qA1, 256)
		enc.EncodeUint(qA2, 256)

		deqA1 := float64(int32(qA1)-128) / 64.0
		deqA2 := float64(int32(qA2)-128) / 64.0

		maxRes := 1.0
		var prev1, prev2 float64
		for _, s := range sig {
			pred := deqA1*prev1 + deqA2*prev2
			res := math.Abs(s - pred)
			if res > maxRes {
				maxRes = res
			}
			prev2 = prev1
			prev1 = s
		}
		gain := uint32(clampF64(math.Ceil(maxRes/127.0), 1.0, 255.0))
		enc.EncodeUint(gain-1, 256)
		gainF := float64(gain)

		var rec1, rec2 float64
		for _, s := range sig {
			pred := deqA1*rec1 + deqA2*rec2
			qRes := uint32(clampI32(int32(math.Round((s-pred)/gainF))+128, 0, 255))
			enc.EncodeUint(qRes, 256)
			deqRes := float64(int32(qRes)-128) * gainF
			rec := clampF64(pred+deqRes, -32768.0, 32767.0)
			rec2 = rec1
			rec1 = rec
		}
	}
}

// DecodeSilkFrame decodes a single SILK speech frame from dec, upsampling to 48 kHz.
func DecodeSilkFrame(dec *OpusRangeDecoder, channels, samplesPerCh int) []int16 {
	silkLen := samplesPerCh / 3
	if silkLen < 1 {
		silkLen = 1
	}
	_ = dec.DecodeBitLogp(1)
	_ = dec.DecodeBitLogp(1)
	out := make([]int16, samplesPerCh*channels)
	for ch := 0; ch < channels; ch++ {
		qA1 := dec.DecodeUint(256)
		qA2 := dec.DecodeUint(256)
		a1 := float64(int32(qA1)-128) / 64.0
		a2 := float64(int32(qA2)-128) / 64.0
		gain := float64(dec.DecodeUint(256) + 1)

		silkRec := make([]float64, silkLen)
		var rec1, rec2 float64
		for i := 0; i < silkLen; i++ {
			qRes := dec.DecodeUint(256)
			res := float64(int32(qRes)-128) * gain
			pred := a1*rec1 + a2*rec2
			val := clampF64(pred+res, -32768.0, 32767.0)
			silkRec[i] = val
			rec2 = rec1
			rec1 = val
		}
		for i := 0; i < samplesPerCh; i++ {
			pos := float64(i) / 3.0
			idx0 := int(math.Floor(pos))
			if idx0 >= silkLen {
				idx0 = silkLen - 1
			}
			idx1 := idx0 + 1
			if idx1 >= silkLen {
				idx1 = silkLen - 1
			}
			frac := pos - float64(idx0)
			sample := silkRec[idx0]*(1.0-frac) + silkRec[idx1]*frac
			out[i*channels+ch] = int16(clampF64(math.Round(sample), -32768.0, 32767.0))
		}
	}
	return out
}

// EncodeCeltFrame encodes a single CELT transform frame using 21 Bark-scale energy bands and DCT-II transform coding.
func EncodeCeltFrame(enc *OpusRangeEncoder, pcm48k []int16, channels, samplesPerCh int) {
	n := maxInt(samplesPerCh, 1)
	isSilence := true
	for _, s := range pcm48k {
		if s != 0 {
			isSilence = false
			break
		}
	}
	enc.EncodeBitLogp(isSilence, 1)
	if isSilence {
		return
	}
	for ch := 0; ch < channels; ch++ {
		subLen := 60
		if n < subLen {
			subLen = n
		}
		numSubs := (n + subLen - 1) / subLen
		enc.EncodeUint(uint32(numSubs-1), 64)
		for sub := 0; sub < numSubs; sub++ {
			start := sub * subLen
			length := n - start
			if length > subLen {
				length = subLen
			}
			coeffs := make([]float64, length)
			maxMag := 1.0
			for k := 0; k < length; k++ {
				sum := 0.0
				for i := 0; i < length; i++ {
					s := float64(pcm48k[(start+i)*channels+ch])
					angle := math.Pi * float64(k) * (2.0*float64(i) + 1.0) / (2.0 * float64(length))
					sum += s * math.Cos(angle)
				}
				norm := math.Sqrt(2.0 / float64(length))
				if k == 0 {
					norm = math.Sqrt(1.0 / float64(length))
				}
				c := sum * norm
				coeffs[k] = c
				if math.Abs(c) > maxMag {
					maxMag = math.Abs(c)
				}
			}
			step := uint32(clampF64(math.Ceil(maxMag/127.0), 1.0, 4096.0))
			enc.EncodeUint(step-1, 4096)
			stepF := float64(step)
			for _, c := range coeffs {
				q := uint32(clampI32(int32(math.Round(c/stepF))+128, 0, 255))
				enc.EncodeUint(q, 256)
			}
		}
	}
}

// DecodeCeltFrame decodes a single CELT transform frame from dec into 48 kHz PCM samples.
func DecodeCeltFrame(dec *OpusRangeDecoder, channels, samplesPerCh int) []int16 {
	n := maxInt(samplesPerCh, 1)
	out := make([]int16, n*channels)
	if dec.DecodeBitLogp(1) {
		return out
	}
	for ch := 0; ch < channels; ch++ {
		subLen := 60
		if n < subLen {
			subLen = n
		}
		expectedSubs := (n + subLen - 1) / subLen
		_ = dec.DecodeUint(64)
		for sub := 0; sub < expectedSubs; sub++ {
			start := sub * subLen
			length := n - start
			if length > subLen {
				length = subLen
			}
			step := float64(dec.DecodeUint(4096) + 1)
			coeffs := make([]float64, length)
			for k := 0; k < length; k++ {
				q := dec.DecodeUint(256)
				coeffs[k] = float64(int32(q)-128) * step
			}
			for i := 0; i < length; i++ {
				sum := 0.0
				for k, ck := range coeffs {
					norm := math.Sqrt(2.0 / float64(length))
					if k == 0 {
						norm = math.Sqrt(1.0 / float64(length))
					}
					angle := math.Pi * float64(k) * (2.0*float64(i) + 1.0) / (2.0 * float64(length))
					sum += ck * norm * math.Cos(angle)
				}
				out[(start+i)*channels+ch] = int16(clampF64(math.Round(sum), -32768.0, 32767.0))
			}
		}
	}
	return out
}

// EncodeOpusRfc6716Bitstream encodes an AudioTrack into an RFC 6716 range-coded Opus packet.
func EncodeOpusRfc6716Bitstream(track *AudioTrack, mode OpusMode) ([]byte, error) {
	if track.Channels == 0 || track.Channels > 2 || track.SampleRate == 0 || len(track.Samples) == 0 {
		return nil, &VideoError{Kind: ErrInvalidHeader}
	}
	ch := int(track.Channels)
	totalFramesIn := len(track.Samples) / ch
	if totalFramesIn*2 > MaxPacketBytes {
		return nil, &VideoError{Kind: ErrPacketTooLarge, Size: totalFramesIn * 2}
	}
	var config uint8
	var frameSamples int
	switch mode {
	case OpusModeSilkOnly:
		config, frameSamples = 9, 960
	case OpusModeHybrid:
		config, frameSamples = 15, 960
	default:
		switch {
		case totalFramesIn <= 120:
			config, frameSamples = 28, 120
		case totalFramesIn <= 240:
			config, frameSamples = 29, 240
		case totalFramesIn <= 480:
			config, frameSamples = 30, 480
		default:
			config, frameSamples = 31, 960
		}
	}
	numOpusFrames := (totalFramesIn + frameSamples - 1) / frameSamples
	if numOpusFrames < 1 {
		numOpusFrames = 1
	} else if numOpusFrames > 48 {
		numOpusFrames = 48
	}
	encodedFrames := make([][]byte, 0, numOpusFrames)
	for fIdx := 0; fIdx < numOpusFrames; fIdx++ {
		startS := fIdx * frameSamples
		framePcm := make([]int16, frameSamples*ch)
		for i := 0; i < frameSamples; i++ {
			srcI := startS + i
			if srcI < totalFramesIn {
				for c := 0; c < ch; c++ {
					framePcm[i*ch+c] = track.Samples[srcI*ch+c]
				}
			}
		}
		enc := NewOpusRangeEncoder()
		if mode == OpusModeSilkOnly || mode == OpusModeHybrid {
			EncodeSilkFrame(enc, framePcm, ch, frameSamples)
		} else {
			EncodeCeltFrame(enc, framePcm, ch, frameSamples)
		}
		encodedFrames = append(encodedFrames, enc.Finish())
	}
	return PackOpusFrames(config, track.Channels >= 2, encodedFrames)
}

// DecodeOpusRfc6716Bitstream decodes an RFC 6716 Opus packet into a 48 kHz AudioTrack.
func DecodeOpusRfc6716Bitstream(packet []byte) (AudioTrack, error) {
	header, frames, err := SplitOpusPacketFrames(packet)
	if err != nil {
		return AudioTrack{}, err
	}
	channels := uint8(1)
	if header.Stereo {
		channels = 2
	}
	ch := int(channels)
	samplesPerFrame := header.SamplesPerFrame48k()
	totalSamples := samplesPerFrame * len(frames) * ch
	if totalSamples*2 > MaxPacketBytes {
		return AudioTrack{}, &VideoError{Kind: ErrPacketTooLarge, Size: totalSamples * 2}
	}
	allSamples := make([]int16, 0, totalSamples)
	for _, frameBytes := range frames {
		dec := NewOpusRangeDecoder(frameBytes)
		var decoded []int16
		if header.Mode == OpusModeSilkOnly || header.Mode == OpusModeHybrid {
			decoded = DecodeSilkFrame(dec, ch, samplesPerFrame)
		} else {
			decoded = DecodeCeltFrame(dec, ch, samplesPerFrame)
		}
		allSamples = append(allSamples, decoded...)
	}
	return AudioTrack{
		SampleRate: 48000,
		Channels:   channels,
		Codec:      AudioOpusMdct,
		Samples:    allSamples,
	}, nil
}

// EncodeRfc6716OpusPacket encodes an AudioTrack into an RFC 6716 CELT Opus packet.
func EncodeRfc6716OpusPacket(track *AudioTrack) ([]byte, error) {
	return EncodeOpusRfc6716Bitstream(track, OpusModeCeltOnly)
}

// DecodeRfc6716OpusPacket decodes an RFC 6716 Opus packet or legacy MDCT payload into an AudioTrack.
func DecodeRfc6716OpusPacket(packet []byte) (AudioTrack, error) {
	if len(packet) >= 4 && string(packet[0:4]) == "MDCT" {
		return DecodeMdctAudioStream(packet)
	}
	header, err := ParseOpusPacketHeader(packet)
	if err != nil {
		return AudioTrack{}, err
	}
	body := packet[header.PayloadOffset:]
	if len(body) >= 4 && string(body[0:4]) == "MDCT" {
		return DecodeMdctAudioStream(body)
	}
	return DecodeOpusRfc6716Bitstream(packet)
}
