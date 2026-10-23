# Security Policy

## Proof of Concept (PoC) Notice

`flux-video` is a standalone **Proof of Concept (PoC)** GPU-accelerated `NV12`/BT.709 video encoder/decoder, ISOBMFF (`.mp4`) container muxer/demuxer, and ISO/IEC 23001-7 Common Encryption (`CENC`) + W3C Encrypted Media Extensions (`EME`) library authored from scratch by **Buntos** as part of the [**Flux Browser**](https://github.com/buntatoes/flux-browser) engine ecosystem.

---

## Supported Versions

| Version | Supported |
| :--- | :---: |
| `v.03` (Latest) | Yes |
| `< v.03` | No |

---

## Reporting a Vulnerability

Please **do not** report security vulnerabilities through public GitHub issues or pull requests.

- **GitHub Private Vulnerability Reporting (Recommended)**: [`https://github.com/buntatoes/flux-video/security/advisories/new`](https://github.com/buntatoes/flux-video/security/advisories/new)
- **Central Ecosystem Security Portal**: [`https://github.com/buntatoes/flux-browser/security/advisories/new`](https://github.com/buntatoes/flux-browser/security/advisories/new)
- **Author Attribution**: **Buntos** ([@buntatoes](https://github.com/buntatoes) · `Buntos <324143321+buntatoes@users.noreply.github.com>`)

*(Note: `324143321+buntatoes@users.noreply.github.com` is a GitHub privacy `noreply` commit-attribution address and does not accept direct inbound email; please use GitHub Private Vulnerability Reporting above to contact the maintainer privately.)*

---

## Security Controls in `flux-video`

- **4K Resolution & Frame Ceilings (`validate_video_dimensions`)**:
  - Enforces `MAX_VIDEO_WIDTH = 4096`, `MAX_VIDEO_HEIGHT = 2160` (`8,847,360` pixels per frame), and `MAX_VIDEO_FRAMES = 36,000` prior to surface allocation.
- **ISOBMFF / `.mp4` Atom-Bomb & Container Guards (`inspect_mp4_metadata`)**:
  - Enforces `MAX_PACKET_BYTES = 16 MiB`, `MAX_CONTAINER_BYTES = 64 MiB`, `MAX_MP4_BOXES = 256`, and `MAX_MP4_BOX_DEPTH = 16` when parsing untrusted `.mp4` containers.
