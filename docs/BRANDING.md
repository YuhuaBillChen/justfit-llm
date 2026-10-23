# JustFit naming and upstream attribution

The product is **JustFit** and its Python distribution is **justfit-llm**.
`justfit serve` is the user entry point; `justfit-llm` is an equivalent CLI alias.

| Component | Name |
| --- | --- |
| Native executable | `source/build/justfit-engine` |
| Main GPU library | `source/build/justfit.metallib` |
| Verification GPU library | `source/build/justfit-verify16.metallib` |
| Native static library | `source/build/engine/libjustfit.a` |
| C++ namespace | `justfit` |
| Shared host/Metal type prefix | `JustFit` |
| Environment variables and build macros | `JUSTFIT_` |
| Prometheus metric prefix | `justfit_` |
| Source launcher and completions | `source/justfit`, `_justfit`, `justfit.bash`, `justfit.fish` |

Use the new environment variable names in custom configurations. For example,
`JUSTFIT_API_KEY`, `JUSTFIT_PORT` and `JUSTFIT_CRASH_TRACE` configure the native
frontend and source launcher. Old product-prefixed environment variables are
not aliases. Update external monitoring queries to the `justfit_` metric names.

The source-level launcher is a developer interface with its own server options.
The Python `justfit serve` command is the reference-preset installation path.
Read each command's help rather than interchanging their flags.

New default data, logs, temporary files and generated caches use JustFit names.
This release does not rename or delete existing upstream data. Generated native
runtime bundles must be rebuilt against the new source manifest; a filename-only
rename of an older binary does not qualify it for this source tree.

## Acknowledgments and compatibility

JustFit derives from Splash 1.3.0. Retain the Apache 2.0 license, copyright notices,
applicable third-party notices and the [project NOTICE](../NOTICE).
Modified upstream source files identify their JustFit modifications.

Published model repository names, such as `incoai/Qwen3.8-27B-Splash`, remain
unchanged so downloads resolve to the actual upstream assets. The existing
`splash-packed-q4`, `splash-packed-q4-moe` and packed-manifest hash identifiers
remain unchanged so existing weights can be read and verified. Binary wire and
file-format magic values are compatibility identifiers, not product names.
Historical upstream measurements and citations keep their original attribution;
they are not relabeled as JustFit measurements.
