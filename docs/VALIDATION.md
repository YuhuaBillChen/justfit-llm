# Validation and limitations

Validation has separate scopes: source integrity, Python tests, installation,
native execution, and device-specific ANE behavior. A pass in one scope does not
certify the others. Use the external acceptance receipt whose SHA256 values match
the exact artifacts you install; receipts from an earlier version are historical.

## Local checks

With the pinned dependencies and a C++20 compiler installed, run:

```bash
python validate_package.py
python -m unittest discover -s tests -v
python -m unittest discover -s service -p 'test_*.py' -v
```

The source and service suites cover routing, startup, cancellation, provisioning,
thinking-budget boundaries, image parsing, and management request protection.
Test counts and actual results belong in the generated acceptance receipt.

For wheel and sdist lifecycle checks, use `scripts/qualify_pip.py` as described in
[Python distribution](PYPI.md). Its local file index does not test TestPyPI or PyPI.

## Apple silicon checks

Build from a fresh installation, then run the text/image examples in the
[quickstart](../QUICKSTART.md), the [thinking-budget checker](THINKING_BUDGET.md),
and the [ANE provisioning and route checks](ANE_PROVISIONING.md).
Keep the Python frontend and native protocol version paired. Record executable,
Metal library, helper, model and profile identities with each device result.

Verify actual ANE execution rather than merely requesting an ANE route. Check
both 4K/8K tiers, phase eligibility, memory denial and GPU fallback, cancellation,
released leases/payloads, and a successful subsequent request. On a language-ANE
configuration, also verify that language execution resumes after image work.

For persistence, prepare a history, restart the service, and inspect the first
request's disk KV reads, composite-state hit, reused/recomputed tokens and output.
A warm second request is not an independent cold-restart observation. Configure
disk quotas and free-space guards for your workload; a payload quota is not an
exact cap on filesystem allocation.

## Interpretation

Synthetic image answers validate selected OCR/routing cases, not broad vision
quality. Identical output is not necessarily a correct answer. Small task sets,
short smoke tests and passing package checks do not establish quality equivalence,
universal acceleration, long-duration reliability or absence of vulnerabilities.
Model weights and compiled assets are separate inputs with their own licenses
and device compatibility. Optional broker adapters require separate integration
tests; direct-engine checks cannot certify an external request-rewriting layer.


## Persistent-state memory regression

After building on Apple silicon, run from `source/`:

```bash
make build/engine-tests/qwen-state-storage
build/engine-tests/qwen-state-storage build/justfit.metallib
build/engine-tests/qwen-state-storage build/justfit.metallib full
```

These tests check immutable-buffer ownership until IO completes, byte-exact
state restore, cancellation/drain, quota refusal and direct disk recovery
without allocating another state snapshot. Full mode uses a 187 MiB state.
They test storage lifetime; they do not establish model-quality equivalence.

The reference node presets use a 22,015 MiB native memory budget. This budget
has been tested for two warm-prefix requests, each with 131,072 input plus
32,768 output tokens (327,680 retained tokens in total). This is an aggregate
B2 capacity result, not a single-request context window or cold-prefill promise.
Capacity and throughput use separate workloads; the prior paired 16K throughput
cohort used a 21,631 MiB budget. Validate memory pressure for local workloads,
including image and concurrent work. A 32 GiB machine with 8 GiB locked away
approximates available RAM but is not identical to 24 GiB hardware.

For a more conservative local configuration, retain an unmodified workspace for
integrity verification, copy it into a separate workspace, and change the
`--max-memory` value in the selected `config/nodes/*.json` to `21631M`.
Integrity verification will correctly report that local preset modification.
The 320K capacity qualification does not apply to that lower budget.
