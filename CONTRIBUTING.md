# Contributing

Build and test from this repository:

```bash
cd rust/frontend && cargo test --release --no-fail-fast
justfit build --project <workspace>   # native runtime; needs Apple's Metal tools
```

On macOS 27 with Xcode, `cargo test` in 2.6.0 passes 198 tests; nine contract
tests still assert pre-2.6 defaults and fixtures and fail. They are tracked for
2.6.1.

When reporting a reproducible issue, include the chip, memory size, macOS
build, the `justfit --version` output, the model revision and the command you
ran. Leave out credentials, private prompts and model weights. Runtime changes
need focused behavioral tests; accelerator changes also need measurements on
the affected chip.

Report JustFit issues at https://github.com/YuhuaBillChen/justfit-llm/issues.
Upstream Splash maintainers are not the support contact for this modified
project.
