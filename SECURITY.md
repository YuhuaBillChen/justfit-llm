# Security and operational scope

`justfit serve` binds to 127.0.0.1 by default. Before binding to another
address, set `--api-key`, and list the names and browser origins you accept
with `--allowed-host` and `--allowed-origin`; network exposure still needs TLS
and access control in front of it. Request bodies are limited to 128 MiB.

`justfit` runs the inference engine and its native workers under a macOS
sandbox profile without outbound network access. The model, draft and ANE
programs are downloaded by `justfit` itself, pinned by revision and SHA-256.

Report vulnerabilities privately through GitHub's security advisories for this
repository rather than in a public issue.
