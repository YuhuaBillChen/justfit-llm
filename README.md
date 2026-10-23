# JustFit 2.3

Local inference for Apple silicon, with coordinated CPU, GPU and ANE work,
concurrent decoding, persistent prefix caching, and conditional ANE image encoding.
Derived from Splash 1.3.0; upstream attribution and licenses are preserved.

**Release candidate — public publication pending.**
Check the matching artifact receipt for validation status. Installing this package
does not activate services. Paper and website are placeholders.

- [Python wheel installation and acceptance](docs/PYPI.md): after installing the wheel,
  `justfit serve` builds, downloads the model and starts a local server. Public index upload pending.
- [Prebuilt runtime assets](docs/RUNTIME_DISTRIBUTION.md): shared macOS arm64 runtime, source-bound checksums and local qualification.
- [Quickstart](QUICKSTART.md): verify, build, and run against a local model package.
- [中文说明](README.zh-CN.md)
- [Product naming and upstream attribution](docs/BRANDING.md)
- [Configuration](docs/CONFIGURATION.md) and [ANE policy](ANE_POLICY.md)
- [Precompiled ANE downloads and installation](docs/ANE_DOWNLOADS.md)
- [New-machine ANE export, offline compilation and qualification](docs/ANE_PROVISIONING.md)
- [Deployment and rollback](docs/DEPLOYMENT.md)
- [Optional external broker adapters](service/broker-integration/README.md)
- [Hard thinking budget and paired native upgrade](docs/THINKING_BUDGET.md)
- [Validation and limitations](docs/VALIDATION.md)
- [Release notes](CHANGELOG.md), [provenance](ORIGIN.json), [release status](RELEASE_STATUS.json)
- [Paper placeholder](paper/README.md), [website placeholder](website/README.md)

## Included

`source/` contains the native runtime, Metal kernels, Python frontend and model
installer. `vision-tools/` contains grouped ANE serving workers, the portable
4K/8K exporter, offline compilation/profile tooling and helper sources.
`service/` contains the guarded managed service and supervisor. `config/nodes/`
contains portable configuration presets.
`scripts/run_local.py` runs a foreground server with a selected node preset.
`tests/` and the service tests verify routing, cancellation and resource protection.

Preset names identify reference configurations, not automatic hardware detection:

| Preset | GPU A8 prefill / verify | Language ANE FFN prefill | ANE vision eligibility |
| --- | --- | --- | --- |
| m4 | off / off | off; CPU prefill enabled | text prefill or decode |
| m5 | on / on | off | text prefill or decode |
| m6 | on / on | on | decode with no active text prefill |

Vision eligibility still requires a qualified local profile, cached assets and
native memory admission. 4K/8K are image patch capacities, not text context sizes.
Idle requests and larger images use GPU. The goal is to free GPU capacity for
text work; ANE image latency is not guaranteed to be lower.

## Build and assets

Apple silicon, macOS 27.0+, Python 3.12–3.14, Xcode/Metal tools and a compatible
CoreAI SDK are required for the complete source build. See the quickstart for
exact commands. The standalone GPU engine retains a macOS 26.4 source target;
the complete Python build and runtime include CoreAI helpers and require macOS 27.0+.
Model weights, compiled vision graphs and machine-specific
CoreAI caches are provisioned separately and are not in this archive.
The first-run path works without an ANE vision profile; enabling conditional
ANE routes requires the local assets described in the configuration guide.

The [engine README](source/README.md) describes JustFit components.
Historical upstream benchmarks are retained separately with their original
attribution; they are not JustFit measurements.

## License and attribution

See [LICENSE](LICENSE), [NOTICE](NOTICE) and
[third-party notices](source/THIRD_PARTY_NOTICES). Retain existing copyright and
inline notices. Model assets retain their own licenses. Contribution guidance is
in [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If JustFit supports your research, please cite our paper:

Yuhua Chen. *JustFit v2: Long-Context Local LLM Serving with Just-in-Time State Management*. Unpublished manuscript, 2026.

```bibtex
@unpublished{chen2026justfitv2,
  author = {Chen, Yuhua},
  title = {{JustFit v2}: Long-Context Local LLM Serving with Just-in-Time State Management},
  year = {2026},
  note = {Unpublished manuscript}
}
```

[CITATION.cff](CITATION.cff) provides machine-readable citation metadata.
A public paper link and identifier will be added when available.
