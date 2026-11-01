// SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0-OR-Commercial
// Copyright (c) 2026 Buntato
//! Compiles the sibling Go module `../anvil-compress` (`github.com/buntatoes/anvil-compress`)
//! into `OUT_DIR` so `anvil-video` executes the Go compression engine.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn find_go_binary() -> PathBuf {
    if let Ok(go_bin) = env::var("ANVIL_GO_BIN") {
        let p = PathBuf::from(go_bin);
        if p.exists() {
            return p;
        }
    }
    let candidates = [
        r"C:\Program Files\Go\bin\go.exe",
        "/usr/local/go/bin/go",
        "/usr/bin/go",
    ];
    for c in candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return p;
        }
    }
    PathBuf::from("go")
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let go_repo = manifest_dir.join("..").join("anvil-compress");
    println!(
        "cargo:rerun-if-changed={}",
        go_repo.join("go.mod").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        go_repo.join("checksum.go").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        go_repo.join("deflate.go").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        go_repo.join("zlib.go").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        go_repo.join("gzip.go").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        go_repo.join("cmd/anvil-compress/main.go").display()
    );

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let bin_name = if cfg!(windows) {
        "anvil-compress-go.exe"
    } else {
        "anvil-compress-go"
    };
    let bin_path = out_dir.join(bin_name);
    let gocache = out_dir.join("gocache");
    let gotmp = out_dir.join("gotmp");
    let _ = std::fs::create_dir_all(&gocache);
    let _ = std::fs::create_dir_all(&gotmp);

    let go = find_go_binary();
    let status = Command::new(&go)
        .current_dir(&go_repo)
        .env("CGO_ENABLED", "0")
        .env("GOCACHE", &gocache)
        .env("GOTMPDIR", &gotmp)
        .arg("build")
        .arg("-trimpath")
        .arg("-ldflags=-s -w")
        .arg("-buildvcs=false")
        .arg("-o")
        .arg(&bin_path)
        .arg("./cmd/anvil-compress")
        .status()
        .expect("failed to invoke Go compiler for ../anvil-compress");

    assert!(
        status.success() && Path::new(&bin_path).exists(),
        "go build failed for ../anvil-compress"
    );

    println!(
        "cargo:rustc-env=ANVIL_COMPRESS_GO_BIN={}",
        bin_path.to_string_lossy()
    );
}
