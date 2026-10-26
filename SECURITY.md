# Security

Report problems in private:

https://github.com/buntatoes/anvil-video/security/advisories/new

The browser repository takes the same kind of report:

https://github.com/buntatoes/anvil-browser/security/advisories/new

Do not put the details in a public issue.

## What is checked

- Dimensions, pixel count, container size, packet size, and MP4 box depth are capped.
- An encrypted sample without a matching key is not decoded.
- AES-128-CTR itself is in `anvil-cipher`. A wrong key does not fail a tag check, because CTR has no tag. The caller has to know the key.

## What is not here

- No hardware decoder. The Direct3D and VA-API names are labels.
- No license-server CDM. Keys in `CdmSession` are local.

Version 0.7.2 is current.
