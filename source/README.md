<!-- Modified for JustFit: product identity and integration. -->
# JustFit Engine

The native engine, Metal kernels and Python frontend used by JustFit.

Start with the [JustFit README](../README.md), [quickstart](../QUICKSTART.md)
and [Python installation guide](../docs/PYPI.md). The supported entry point is
`justfit serve`; it selects a reference preset and starts a foreground local server.

## Components

- `build/justfit-engine`: native inference engine.
- `build/justfit.metallib`: GPU kernels.
- `build/justfit-verify16.metallib`: speculative verification kernels.
- `server/`: HTTP frontend, streaming and protocol handling.
- `install/`: model acquisition and the lower-level source launcher `./justfit`.

For the complete source build, run `justfit build` in a verified workspace as
shown in the quickstart. The source-level `make` targets are developer interfaces;
they do not replace the complete build and qualification process.

[Developer reference](DEVELOPMENT.md) · [Security](../SECURITY.md) ·
[Contributing](../CONTRIBUTING.md)

## Acknowledgments

JustFit derives from [Splash 1.3.0](https://github.com/incoai/splash).
We thank its authors and contributors. See the [project NOTICE](../NOTICE),
[Apache 2.0 license](LICENSE), [third-party notices](THIRD_PARTY_NOTICES) and
[pinned provenance](../ORIGIN.json). Modified source files carry modification notices.
The retained [upstream performance record](docs/performance.md) describes Splash
experiments and is not a JustFit benchmark or a claim of upstream endorsement.
