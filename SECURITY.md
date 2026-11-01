# Security

Report problems in private:

https://github.com/buntatoes/anvil-video/security/advisories/new

Do not put the details in a public issue.

## What is checked

- Dimensions, pixel count, container size, packet size, and MP4 box depth are capped.
- An encrypted sample without a matching key is not decoded.
- AES-128-CTR CENC sample encryption is implemented in pure Go (`cenc_encrypt_sample` / `CencEncryptSample`). A wrong key does not fail a tag check, because CTR has no tag. The caller has to know the key.

## What is not here

- No hardware decoder. The Direct3D and VA-API names are labels.
- No license-server CDM. Keys in `CdmSession` are local.

Version 0.11.0 is current.
