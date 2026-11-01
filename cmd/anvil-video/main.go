// SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0-OR-Commercial
// Copyright (c) 2026 Buntato. Verify canonical repository history at https://github.com/buntatoes

package main

import (
	"fmt"
	"io"
	"os"

	anvilvideo "github.com/buntatoes/anvil-video"
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr))
}

func run(args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, "usage: anvil-video <self-test|probe-caps|inspect-mp4|inspect-webm|inspect-fmp4>")
		return 2
	}
	switch args[0] {
	case "self-test":
		if err := runSelfTest(); err != nil {
			fmt.Fprintf(stderr, "self-test failed: %v\n", err)
			return 1
		}
		fmt.Fprintf(stdout, "anvil-video %s ok\n", anvilvideo.Version)
		return 0
	case "probe-caps":
		caps := anvilvideo.ProbeHardwareVideoCapabilities()
		fmt.Fprintf(stdout, "gpu_ready=%v mf_present=%v adapter=%q profiles=%d\n",
			caps.GpuCompositorReady,
			caps.PlatformMediaFoundationPresent,
			caps.AdapterDescription,
			len(caps.SupportedProfiles),
		)
		return 0
	case "inspect-mp4":
		data, err := io.ReadAll(io.LimitReader(stdin, int64(anvilvideo.MaxContainerBytes)+1))
		if err != nil {
			fmt.Fprintf(stderr, "read stdin: %v\n", err)
			return 1
		}
		meta, err := anvilvideo.InspectMp4Metadata(data)
		if err != nil {
			fmt.Fprintf(stderr, "inspect-mp4: %v\n", err)
			return 1
		}
		fmt.Fprintf(stdout, "brand=%s width=%d height=%d duration_ms=%d audio=%v\n",
			string(meta.Brand[:]), meta.Width, meta.Height, meta.DurationMs, meta.HasAudioTrack)
		return 0
	case "inspect-webm":
		data, err := io.ReadAll(io.LimitReader(stdin, int64(anvilvideo.MaxContainerBytes)+1))
		if err != nil {
			fmt.Fprintf(stderr, "read stdin: %v\n", err)
			return 1
		}
		meta, err := anvilvideo.InspectWebmMetadata(data)
		if err != nil {
			fmt.Fprintf(stderr, "inspect-webm: %v\n", err)
			return 1
		}
		fmt.Fprintf(stdout, "doctype=%s codec=%s width=%d height=%d clusters=%d blocks=%d\n",
			meta.DocType, meta.VideoCodecID, meta.Width, meta.Height, meta.ClusterCount, meta.BlockCount)
		return 0
	case "inspect-fmp4":
		data, err := io.ReadAll(io.LimitReader(stdin, int64(anvilvideo.MaxContainerBytes)+1))
		if err != nil {
			fmt.Fprintf(stderr, "read stdin: %v\n", err)
			return 1
		}
		info, err := anvilvideo.InspectFmp4Segment(data)
		if err != nil {
			fmt.Fprintf(stderr, "inspect-fmp4: %v\n", err)
			return 1
		}
		fmt.Fprintf(stdout, "init=%v media=%v seq=%d samples=%d duration_ms=%d\n",
			info.HasInitSegment, info.HasMediaFragment, info.SequenceNumber, info.SampleCount, info.FragmentDurationMs)
		return 0
	default:
		fmt.Fprintf(stderr, "unknown command: %s\n", args[0])
		return 2
	}
}

func makeSolidFrame(w, h, ptsMs uint32, keyframe bool, r, g, b byte) (anvilvideo.VideoFrame, error) {
	rgba := make([]byte, int(w*h*4))
	for i := 0; i < len(rgba); i += 4 {
		rgba[i] = r
		rgba[i+1] = g
		rgba[i+2] = b
		rgba[i+3] = 255
	}
	return anvilvideo.NewVideoFrame(w, h, ptsMs, keyframe, rgba)
}

func runSelfTest() error {
	f0, err := makeSolidFrame(16, 16, 0, true, 40, 120, 200)
	if err != nil {
		return err
	}
	f1, err := makeSolidFrame(16, 16, 40, false, 48, 124, 196)
	if err != nil {
		return err
	}
	frames := []anvilvideo.VideoFrame{f0, f1}

	dctStream, err := anvilvideo.EncodeDctMcBitstream(anvilvideo.ProfileH264High, 25, 16, frames)
	if err != nil {
		return err
	}
	decodedDct, err := anvilvideo.DecodeDctMcBitstream(dctStream, false)
	if err != nil || decodedDct.Fps != 25 || len(decodedDct.Frames) != 2 {
		return fmt.Errorf("dct roundtrip mismatch: err=%v fps=%d len=%d", err, decodedDct.Fps, len(decodedDct.Frames))
	}

	vp8Stream, err := anvilvideo.EncodeVp8Rfc6386Stream(25, 16, frames)
	if err != nil {
		return err
	}
	decodedVp8, err := anvilvideo.DecodeVp8Rfc6386Stream(vp8Stream, false)
	if err != nil || len(decodedVp8.Frames) != 2 {
		return fmt.Errorf("vp8 roundtrip mismatch: %v", err)
	}

	vp9Stream, err := anvilvideo.EncodeVp9Stream(25, 16, frames)
	if err != nil {
		return err
	}
	decodedVp9, err := anvilvideo.DecodeVp9Stream(vp9Stream, false)
	if err != nil || len(decodedVp9.Frames) != 2 {
		return fmt.Errorf("vp9 roundtrip mismatch: %v", err)
	}

	av1Stream, err := anvilvideo.EncodeAv1ObuStream(25, 16, frames)
	if err != nil {
		return err
	}
	decodedAv1, err := anvilvideo.DecodeAv1ObuStream(av1Stream, false)
	if err != nil || len(decodedAv1.Frames) != 2 {
		return fmt.Errorf("av1 roundtrip mismatch: %v", err)
	}

	pcm := make([]int16, 960)
	for i := range pcm {
		pcm[i] = int16((i * 97) % 8000)
	}
	track := &anvilvideo.AudioTrack{
		SampleRate: 48000,
		Channels:   1,
		Codec:      anvilvideo.AudioOpusMdct,
		Samples:    pcm,
	}
	opusPkt, err := anvilvideo.EncodeRfc6716OpusPacket(track)
	if err != nil {
		return err
	}
	decOpus, err := anvilvideo.DecodeRfc6716OpusPacket(opusPkt)
	if err != nil || len(decOpus.Samples) == 0 {
		return fmt.Errorf("opus roundtrip mismatch: %v", err)
	}
	return nil
}
