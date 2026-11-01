// SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0-OR-Commercial
// Copyright (c) 2026 Buntato
//! IPC bridge to the standalone Go `anvil-compress` (`v.04`) engine binary
//! (`github.com/buntatoes/anvil-compress/cmd/anvil-compress --worker`).

use std::fmt;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Mutex, OnceLock};

/// Default maximum decompressed payload size (32 MiB).
pub const DEFAULT_MAX_DECOMPRESSED_BYTES: usize = 32 * 1024 * 1024;

const OP_DEFLATE_DECOMPRESS: u8 = 1;
const OP_DEFLATE_COMPRESS: u8 = 2;
const OP_ZLIB_DECOMPRESS: u8 = 3;
const OP_ZLIB_COMPRESS: u8 = 4;
const OP_GZIP_DECOMPRESS: u8 = 5;
const OP_GZIP_COMPRESS: u8 = 6;
const OP_DECOMPRESS_AUTO: u8 = 7;
const OP_ADLER32: u8 = 8;
const OP_CRC32: u8 = 9;

static EMBEDDED_GO_BIN: &[u8] = include_bytes!(env!("ANVIL_COMPRESS_GO_BIN"));

/// Decompression or framing failure returned by the Go `anvil-compress` engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompressError {
    UnexpectedEof,
    InvalidHeader(&'static str),
    InvalidBlockType(u8),
    StoredLengthMismatch,
    InvalidHuffmanTree(&'static str),
    InvalidCode,
    InvalidDistance(u16),
    ChecksumMismatch { expected: u32, actual: u32 },
    LengthMismatch { expected: u32, actual: u32 },
    OutputLimitExceeded { limit: usize },
}

impl fmt::Display for CompressError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => write!(f, "unexpected end of compressed stream"),
            Self::InvalidHeader(msg) => write!(f, "invalid header: {msg}"),
            Self::InvalidBlockType(bt) => write!(f, "invalid DEFLATE block type {bt}"),
            Self::StoredLengthMismatch => write!(f, "stored DEFLATE block LEN/NLEN mismatch"),
            Self::InvalidHuffmanTree(msg) => write!(f, "invalid Huffman code tree: {msg}"),
            Self::InvalidCode => write!(f, "invalid Huffman symbol in bitstream"),
            Self::InvalidDistance(d) => write!(f, "invalid LZ77 back-reference distance {d}"),
            Self::ChecksumMismatch { expected, actual } => write!(
                f,
                "checksum mismatch (expected 0x{expected:08x}, got 0x{actual:08x})"
            ),
            Self::LengthMismatch { expected, actual } => write!(
                f,
                "uncompressed length mismatch (expected {expected}, got {actual})"
            ),
            Self::OutputLimitExceeded { limit } => {
                write!(f, "decompressed output exceeds {limit} byte cap")
            }
        }
    }
}

impl std::error::Error for CompressError {}

struct GoWorker {
    _child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
}

fn resolve_go_binary_path() -> PathBuf {
    let ext = if cfg!(windows) { ".exe" } else { "" };
    let tmp_path = std::env::temp_dir().join(format!(
        "anvil-compress-go-v04-{}{}",
        std::process::id(),
        ext
    ));
    if !tmp_path.exists() {
        let _ = std::fs::write(&tmp_path, EMBEDDED_GO_BIN);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&tmp_path, std::fs::Permissions::from_mode(0o755));
        }
    }
    tmp_path
}

fn spawn_worker() -> Result<GoWorker, CompressError> {
    let bin = resolve_go_binary_path();
    let mut cmd = Command::new(bin);
    cmd.arg("--worker")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let mut child = cmd
        .spawn()
        .map_err(|_| CompressError::InvalidHeader("failed to start Go anvil-compress worker"))?;
    let stdin = child
        .stdin
        .take()
        .ok_or(CompressError::InvalidHeader("missing worker stdin"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or(CompressError::InvalidHeader("missing worker stdout"))?;
    Ok(GoWorker {
        _child: child,
        stdin,
        stdout,
    })
}

fn worker_slot() -> &'static Mutex<Option<GoWorker>> {
    static WORKER: OnceLock<Mutex<Option<GoWorker>>> = OnceLock::new();
    WORKER.get_or_init(|| Mutex::new(None))
}

/// Pre-spawns the persistent Go `anvil-compress` worker process if not already running.
pub fn warmup_worker() {
    if let Ok(mut guard) = worker_slot().lock()
        && guard.is_none()
    {
        *guard = spawn_worker().ok();
    }
}

fn read_worker_error(stdout: &mut ChildStdout) -> CompressError {
    let mut hdr = [0u8; 13];
    if stdout.read_exact(&mut hdr).is_err() {
        return CompressError::UnexpectedEof;
    }
    let kind = hdr[0];
    let expected = u32::from_le_bytes([hdr[1], hdr[2], hdr[3], hdr[4]]);
    let actual = u32::from_le_bytes([hdr[5], hdr[6], hdr[7], hdr[8]]);
    let limit = u32::from_le_bytes([hdr[9], hdr[10], hdr[11], hdr[12]]) as usize;
    let mut dlen_buf = [0u8; 4];
    if stdout.read_exact(&mut dlen_buf).is_ok() {
        let dlen = u32::from_le_bytes(dlen_buf) as usize;
        let mut detail = vec![0u8; dlen];
        let _ = stdout.read_exact(&mut detail);
    }
    match kind {
        0 => CompressError::UnexpectedEof,
        1 => CompressError::InvalidHeader("invalid compressed stream header"),
        2 => CompressError::InvalidBlockType(actual as u8),
        3 => CompressError::StoredLengthMismatch,
        4 => CompressError::InvalidHuffmanTree("invalid dynamic Huffman tree"),
        5 => CompressError::InvalidCode,
        6 => CompressError::InvalidDistance(actual as u16),
        7 => CompressError::ChecksumMismatch { expected, actual },
        8 => CompressError::LengthMismatch { expected, actual },
        9 => CompressError::OutputLimitExceeded { limit },
        _ => CompressError::InvalidHeader("compression error"),
    }
}

fn call_decompress(op: u8, input: &[u8], max_out: usize) -> Result<Vec<u8>, CompressError> {
    let mut guard = worker_slot()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if guard.is_none() {
        *guard = Some(spawn_worker()?);
    }
    let w = guard.as_mut().unwrap();
    let mut req = [0u8; 9];
    req[0] = op;
    let capped_max = max_out.min(u32::MAX as usize) as u32;
    req[1..5].copy_from_slice(&capped_max.to_le_bytes());
    req[5..9].copy_from_slice(&(input.len() as u32).to_le_bytes());
    if w.stdin.write_all(&req).is_err()
        || w.stdin.write_all(input).is_err()
        || w.stdin.flush().is_err()
    {
        *guard = None;
        return Err(CompressError::UnexpectedEof);
    }
    let mut status = [0u8; 1];
    if w.stdout.read_exact(&mut status).is_err() {
        *guard = None;
        return Err(CompressError::UnexpectedEof);
    }
    if status[0] == 0 {
        let mut len_buf = [0u8; 4];
        w.stdout
            .read_exact(&mut len_buf)
            .map_err(|_| CompressError::UnexpectedEof)?;
        let out_len = u32::from_le_bytes(len_buf) as usize;
        let mut out = vec![0u8; out_len];
        w.stdout
            .read_exact(&mut out)
            .map_err(|_| CompressError::UnexpectedEof)?;
        Ok(out)
    } else {
        Err(read_worker_error(&mut w.stdout))
    }
}

fn call_compress(op: u8, input: &[u8]) -> Vec<u8> {
    let mut guard = worker_slot()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if guard.is_none() {
        *guard = spawn_worker().ok();
    }
    let Some(w) = guard.as_mut() else {
        return Vec::new();
    };
    let mut req = [0u8; 5];
    req[0] = op;
    req[1..5].copy_from_slice(&(input.len() as u32).to_le_bytes());
    if w.stdin.write_all(&req).is_err()
        || w.stdin.write_all(input).is_err()
        || w.stdin.flush().is_err()
    {
        *guard = None;
        return Vec::new();
    }
    let mut status = [0u8; 1];
    if w.stdout.read_exact(&mut status).is_err() || status[0] != 0 {
        *guard = None;
        return Vec::new();
    }
    let mut len_buf = [0u8; 4];
    if w.stdout.read_exact(&mut len_buf).is_err() {
        *guard = None;
        return Vec::new();
    }
    let out_len = u32::from_le_bytes(len_buf) as usize;
    let mut out = vec![0u8; out_len];
    if w.stdout.read_exact(&mut out).is_err() {
        *guard = None;
        return Vec::new();
    }
    out
}

fn call_checksum(op: u8, input: &[u8]) -> u32 {
    let mut guard = worker_slot()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if guard.is_none() {
        *guard = spawn_worker().ok();
    }
    let Some(w) = guard.as_mut() else {
        return 0;
    };
    let mut req = [0u8; 5];
    req[0] = op;
    req[1..5].copy_from_slice(&(input.len() as u32).to_le_bytes());
    if w.stdin.write_all(&req).is_err()
        || w.stdin.write_all(input).is_err()
        || w.stdin.flush().is_err()
    {
        *guard = None;
        return 0;
    }
    let mut resp = [0u8; 5];
    if w.stdout.read_exact(&mut resp).is_err() || resp[0] != 0 {
        *guard = None;
        return 0;
    }
    u32::from_le_bytes([resp[1], resp[2], resp[3], resp[4]])
}

pub fn deflate_decompress(input: &[u8]) -> Result<Vec<u8>, CompressError> {
    call_decompress(OP_DEFLATE_DECOMPRESS, input, DEFAULT_MAX_DECOMPRESSED_BYTES)
}

pub fn deflate_decompress_bounded(
    input: &[u8],
    max_output_bytes: usize,
) -> Result<Vec<u8>, CompressError> {
    call_decompress(OP_DEFLATE_DECOMPRESS, input, max_output_bytes)
}

pub fn deflate_compress(input: &[u8]) -> Vec<u8> {
    call_compress(OP_DEFLATE_COMPRESS, input)
}

pub fn zlib_decompress(input: &[u8]) -> Result<Vec<u8>, CompressError> {
    call_decompress(OP_ZLIB_DECOMPRESS, input, DEFAULT_MAX_DECOMPRESSED_BYTES)
}

pub fn zlib_decompress_bounded(
    input: &[u8],
    max_output_bytes: usize,
) -> Result<Vec<u8>, CompressError> {
    call_decompress(OP_ZLIB_DECOMPRESS, input, max_output_bytes)
}

pub fn zlib_compress(input: &[u8]) -> Vec<u8> {
    call_compress(OP_ZLIB_COMPRESS, input)
}

pub fn gzip_decompress(input: &[u8]) -> Result<Vec<u8>, CompressError> {
    call_decompress(OP_GZIP_DECOMPRESS, input, DEFAULT_MAX_DECOMPRESSED_BYTES)
}

pub fn gzip_decompress_bounded(
    input: &[u8],
    max_output_bytes: usize,
) -> Result<Vec<u8>, CompressError> {
    call_decompress(OP_GZIP_DECOMPRESS, input, max_output_bytes)
}

pub fn gzip_compress(input: &[u8]) -> Vec<u8> {
    call_compress(OP_GZIP_COMPRESS, input)
}

pub fn decompress_auto(input: &[u8], max_output_bytes: usize) -> Result<Vec<u8>, CompressError> {
    call_decompress(OP_DECOMPRESS_AUTO, input, max_output_bytes)
}

pub fn adler32(data: &[u8]) -> u32 {
    call_checksum(OP_ADLER32, data)
}

pub fn crc32(data: &[u8]) -> u32 {
    call_checksum(OP_CRC32, data)
}

pub use deflate_compress as compress_deflate;
pub use deflate_decompress as decompress_deflate;
pub use deflate_decompress_bounded as decompress_deflate_bounded;
pub use gzip_compress as compress_gzip;
pub use gzip_decompress as decompress_gzip;
pub use gzip_decompress_bounded as decompress_gzip_bounded;
pub use zlib_compress as compress_zlib;
pub use zlib_decompress as decompress_zlib;
pub use zlib_decompress_bounded as decompress_zlib_bounded;
