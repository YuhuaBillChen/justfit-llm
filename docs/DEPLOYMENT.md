# Managed deployment and rollback

Installing the source archive or Python package does not register or activate a
service. First qualify a local build using the [validation guide](VALIDATION.md).

## Release layout

A managed release contains `source/`, `venv/`, service modules, `service.json`,
`engine-command.json`, `engine-env.json`, `sha256.json`, qualified `profiles/`,
and writable `cache/`, `logs/` and `ane-meter/` directories. Copy the supplied
`service/` modules into the configured release root; do not run an unconfigured
`service/service.py` directly.

`service.json` provides the model directory, `http_port` and `native_port`.
`engine-command.json` is a pinned argv array using `{release}`, `{model}`, `{port}`
and `{cache}` placeholders. `engine-env.json` supplies the chosen execution
settings and local asset paths. `sha256.json` pins static files, excluding writable
cache and log directories. Investigate unexpected hash differences before updating
pins; a matching manifest is not a substitute for qualification.

The managed storage guard uses a 20 GiB free-space threshold and reopens at 24 GiB.
Engine logs are limited to 32 MiB with four backups; ANE diagnostics are limited
to 128 MiB. Review these settings for your installation. Guard actions stop
admission or the owned engine rather than deleting user cache data. The foreground
quickstart does not replace managed-service protection.

## Upgrade sequence

1. Record the existing service entry, argv, environment, artifact identities,
   health and active requests. Retain the old release and cache for rollback.
2. Qualify the new dependencies, executable, shaders, helpers and assets in an
   isolated directory. Protocol 9 requires the matching frontend and executable.
   Preserve existing cache files and test actual cold restore; do not rewrite
   metadata to manufacture compatibility.
3. Prepare the new service entry and its rollback counterpart. Choose the launch
   domain appropriate to your installation. Preserve any external media or broker
   services and verify required power/sleep behavior on portable machines.
4. Drain requests, then replace only the intended service entry/process group.
   Do not terminate unrelated processes to free a port.
5. Test the actual user-facing endpoint: model listing, text, images, streaming,
   cancellation, thinking budgets, ANE eligibility, GPU fallback and recovery.
   If a broker rewrites requests, verify complete image history and tool pairs,
   plus explicit context-limit errors. See the [optional adapters](../service/broker-integration/README.md).
6. Restart and verify first-request disk restoration, output and process health.
   Exercise rollback to the saved release, verify it, then restore the candidate
   and repeat the endpoint checks.
7. Keep a local timestamped deployment receipt with artifact hashes and results.
   Record untested conditions explicitly; package publication does not establish
   the health of an individual installation.
