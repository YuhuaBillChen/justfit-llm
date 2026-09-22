#!/usr/bin/env python3
"""Run one deterministic text-only context probe against a Splash server."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:18998")
    parser.add_argument("--model", default="incoai/Qwen3.8-27B-Splash")
    parser.add_argument("--unit", default=" alpha")
    parser.add_argument("--repetitions", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=7200.0)
    args = parser.parse_args()

    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.unit * args.repetitions}],
        "reasoning_effort": "none",
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        args.base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=args.timeout) as result:
        response = json.load(result)
    content = response["choices"][0]["message"]["content"]
    print(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": {
                    "unit": args.unit,
                    "unit_repetitions": args.repetitions,
                    "requested_output_tokens": args.max_tokens,
                    "reasoning_effort": "none",
                    "temperature": 0,
                },
                "wall_seconds": time.monotonic() - started,
                "output_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "response": response,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
