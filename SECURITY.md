# Security and operational scope

The local quickstart binds to localhost. The optional managed proxy also binds
to loopback and accepts only loopback Host names. It rejects Origin and Fetch
Metadata headers because it is a machine-to-machine endpoint without a browser UI.
Request bodies are limited to 128 MiB, including streamed uploads, with a 30-second
upload deadline before engine acquisition. Local broker load/unload, health/model
polling and streaming remain supported.

Local processes are trusted by default. For an explicit credential boundary, set
`JUSTFIT_MANAGEMENT_API_KEY` in the service environment and configure every local
broker/health client to send `Authorization: Bearer <key>` on every route.
The key must be nonempty visible ASCII. Do not expose this proxy directly to an
untrusted network; network ingress still needs authentication, TLS and access
controls. The main frontend has separate `--api-key`, Host and Origin controls.

The pinned certifi and Pillow versions include certificate-trust and image-parser
security fixes. Regression tests cover the affected CA root, bounded APNG parsing,
and ordinary PNG/GIF/JPEG/WebP inputs. Keep dependencies current; these checks do
not prove absence of other vulnerabilities or qualify untrusted native assets.
Validate upgrades in an isolated environment before changing an installed service.

Do not include credentials, private prompts, caches or model weights in issue reports.
Report JustFit vulnerabilities privately to contact@yuhuachen.com.
Upstream maintainers are not the security contact for this modified project.
