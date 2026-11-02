// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Buntato. Verify canonical repository history at https://github.com/buntatoes

package main

import (
	"bytes"
	"strings"
	"testing"

	anvilvideo "github.com/buntatoes/anvil-video"
)

func TestCLISelfTestAndProbeCaps(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := run([]string{"self-test"}, strings.NewReader(""), &stdout, &stderr)
	if code != 0 {
		t.Fatalf("self-test failed (%d): %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "anvil-video 0.11.0 ok") {
		t.Fatalf("unexpected self-test output: %q", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	code = run([]string{"probe-caps"}, strings.NewReader(""), &stdout, &stderr)
	if code != 0 {
		t.Fatalf("probe-caps failed (%d): %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "profiles=5") {
		t.Fatalf("unexpected probe-caps output: %q", stdout.String())
	}
}

func TestCLIContainerInspections(t *testing.T) {
	mp4Bytes, err := anvilvideo.GenerateDemoMp4(16, 16, 2)
	if err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := run([]string{"inspect-mp4"}, bytes.NewReader(mp4Bytes), &stdout, &stderr)
	if code != 0 || !strings.Contains(stdout.String(), "width=16 height=16") {
		t.Fatalf("inspect-mp4 failed (%d): out=%q err=%q", code, stdout.String(), stderr.String())
	}

	f0, _ := makeSolidFrame(16, 16, 0, true, 50, 100, 150)
	vStream, _ := anvilvideo.EncodeVp8Rfc6386Stream(24, 16, []anvilvideo.VideoFrame{f0})
	webmBytes := anvilvideo.WrapWebmContainer(16, 16, 1000, "V_VP8", vStream)
	stdout.Reset()
	stderr.Reset()
	code = run([]string{"inspect-webm"}, bytes.NewReader(webmBytes), &stdout, &stderr)
	if code != 0 || !strings.Contains(stdout.String(), "doctype=webm") {
		t.Fatalf("inspect-webm failed (%d): out=%q err=%q", code, stdout.String(), stderr.String())
	}

	fmp4Init := anvilvideo.WrapFmp4InitSegment(16, 16, anvilvideo.ProfileH264High, false)
	stdout.Reset()
	stderr.Reset()
	code = run([]string{"inspect-fmp4"}, bytes.NewReader(fmp4Init), &stdout, &stderr)
	if code != 0 || !strings.Contains(stdout.String(), "init=true") {
		t.Fatalf("inspect-fmp4 failed (%d): out=%q err=%q", code, stdout.String(), stderr.String())
	}
}
