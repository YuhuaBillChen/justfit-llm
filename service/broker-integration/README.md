# Optional external broker adapters

The direct JustFit quickstart does not need these patches. They target specific
external broker source layouts and do not install a complete broker. Inspect the
patches and compare the target revision before applying them. For another broker,
implement and test the equivalent behavior instead of applying a patch blindly.

## Preserve image history

`history-preservation.patch` targets `ai_service/broker/request_rewrite.py`.
`BASE_SHA256.json` records the reviewed before/after source identities. It bypasses
legacy approximate image-prompt trimming for the resolved target
`incoai/Qwen3.8-27B-Splash`. Other targets retain their existing behavior; additional
model names require explicit review. The native engine enforces its actual context
limit. Already discarded client history cannot be recovered by the adapter.

From a compatible broker checkout:

```bash
shasum -a 256 ai_service/broker/request_rewrite.py
git apply --check /absolute/path/JustFit-2.3/service/broker-integration/history-preservation.patch
```

Back up the target source, review local changes, and apply only after confirming
compatibility. Drain requests before restarting the broker's managed service.
A source edit alone does not reload a running process. Inspect supervisor behavior:
a child exit may stop other children in the same service group.

Validate the final endpoint with synthetic history:

```bash
python scripts/check_broker_history.py \
  --base-url http://YOUR_NODE:YOUR_PORT --output local-state/broker-history-check
```

The checker includes an early fact, tool-call/result pairing and a screenshot.
Both requests must retrieve the fact and image answer; the follow-up must reuse
at least 90% of prompt tokens. Test explicit over-context rejection separately.
Use synthetic inputs and avoid recording real user conversations. This is a
functional integration check, not a real-workload cache-hit-rate benchmark.

## Health-probe locking

`pool-probe-lock.patch` targets `ai_service/broker/openai_pool_broker.py`. It addresses a
lock cycle between routing and health-probe workers by reading the reservation
ledger under the lock used by its writers. Confirm the target path in the patch
and review the locking discipline in your revision before application.
`test_pool_probe_lock.py` requires the external broker's `ai_service` package;
it is not part of the standalone engine test suite. Run the broker's existing
suite as well as this regression, then verify routing and health under concurrency.

Model aliases can override request-level reasoning settings in an external broker.
Use its base model when testing custom budgets, and confirm that the final request
reaches the native endpoint with the intended control. Budget behavior is described
in [THINKING_BUDGET.md](../../docs/THINKING_BUDGET.md).
