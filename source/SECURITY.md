<!-- Modified for JustFit: product identity and integration. -->
# Security

Please report vulnerabilities privately to <contact@yuhuachen.com> rather than in a
public issue. Include the JustFit version (`justfit --version`), macOS version,
and steps to reproduce. We will acknowledge your report and keep you informed
until it is resolved.

JustFit serves on `127.0.0.1`, and authentication is off by default. Before
exposing the server beyond the local machine, set a key with `JUSTFIT_API_KEY`
or `justfit serve --api-key`; API requests then need it as a bearer token or
`x-api-key`. `--allowed-origin '*'` lets every web page open in a browser that
reaches the server use it, so set a key with it. Crash traces, which JustFit
records only with `JUSTFIT_CRASH_TRACE=1`, can contain private conversation
data. Exposing the server without a key or a proxy is outside the threat model.
