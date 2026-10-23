"""Isolated fixed-shape ANE vision adapter; launched by PreparedVision.

Usage: python ane_vision_worker.py BRIDGE_DIR ASSET INPUT OUTPUT IMAGE_COUNT
No Torch/compiler imports at serving time. Only the pinned bridge and NumPy
are needed. The parent validates the deployment profile and output contract.
"""

import time,json
import os
import sys
from pathlib import Path


def main():
    started = time.perf_counter()
    phase = {}
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("bridge")
    parser.add_argument("asset")
    parser.add_argument("source", nargs="?")
    parser.add_argument("target", nargs="?")
    parser.add_argument("count", nargs="?", type=int)
    parser.add_argument("--grid", type=int, choices=(32,64), default=32)
    parser.add_argument("--entrypoint", default="vision_tower_ane4_safe")
    parser.add_argument("--prewarm", action="store_true")
    parser.add_argument("--serve")
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--positions")
    parser.add_argument("--grids")
    parser.add_argument("--cache-record")
    parser.add_argument("--cached-only", action="store_true")
    args = parser.parse_intermixed_args()
    if args.capacity:
        if not args.cache_record:
            raise ValueError("variable-grid worker requires a compilation record")
        from variable_vision_worker import run
        return run(args)
    bridge, asset = args.bridge, args.asset
    prewarm = args.prewarm
    count = 0 if prewarm else args.count
    source, target = args.source, args.target
    if not prewarm and not args.serve and (count is None or not 1 <= count <= 4 or not source or not target):
        raise ValueError("invalid image count or paths")
    grid = args.grid
    side = grid * 16
    patches_count = grid * grid
    rows_count = patches_count // 4
    os.environ["COREAI_BRIDGE_LIB"] = str(Path(bridge) / "libcoreai_bridge.dylib")
    sys.path.insert(0, bridge)
    import coreai_bridge as B
    import numpy as np

    phase["imports_seconds"] = time.perf_counter() - started
    t = time.perf_counter()
    raw = b"" if prewarm or args.serve else Path(source).read_bytes()
    from ane_model_cache import load_function
    model, function = load_function(B, Path(asset), args.entrypoint, repair_cache=prewarm)
    from ane_placement import require_ane
    require_ane(asset, args.entrypoint)
    inputs = {n: function.buffer("input", n) for n in function.input_names}
    outputs = {n: function.buffer("output", n) for n in function.output_names}
    if inputs["x"].np.shape != (patches_count, 1536) or outputs["y"].np.shape != (rows_count, 5120):
        raise ValueError("unexpected compiled vision signature")
    binding = function.bind(inputs, outputs)
    phase["read_and_load_seconds"] = time.perf_counter() - t
    phase["images"] = count
    phase["patch_seconds"] = 0
    phase["encode_seconds"] = 0
    phase["convert_write_seconds"] = 0
    if prewarm:
        inputs["x"].np[:] = 0
        B.run([binding])
        if not np.isfinite(outputs["y"].np).all():
            raise ValueError("non-finite ANE warmup output")
        return
    def encode(raw, target, count):
        if type(count) is not int or not 1 <= count <= 4 or len(raw) != count * side * side * 3:
            raise ValueError("invalid fixed-shape pixels")
        phase.update(images=count, patch_seconds=0, encode_seconds=0, convert_write_seconds=0)
        with Path(target).open("wb") as result:
            for i in range(count):
                t = time.perf_counter()
                pixels = np.frombuffer(
                    raw, np.uint8, side * side * 3, i * side * side * 3
                ).reshape(side, side, 3)
                normalized = (pixels.astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)
                patches = normalized.reshape(3, grid // 2, 2, 16, grid // 2, 2, 16).transpose(
                    1, 4, 2, 5, 0, 3, 6
                )
                patches = patches.reshape(patches_count, 3, 1, 16, 16)
                inputs["x"].np[:] = np.broadcast_to(patches, (patches_count, 3, 2, 16, 16)).reshape(
                    patches_count, 1536
                )
                phase["patch_seconds"] += time.perf_counter() - t
                t = time.perf_counter()
                B.run([binding])
                phase["encode_seconds"] += time.perf_counter() - t
                t = time.perf_counter()
                values = outputs["y"].np.astype(np.float32)
                if not np.isfinite(values).all():
                    raise ValueError("non-finite ANE output")
                bits = values.view(np.uint32)
                # FP32 -> BF16 round-to-nearest, ties-to-even, not truncation.
                rows = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2")
                result.write(rows.tobytes())
                phase["convert_write_seconds"] += time.perf_counter() - t
    if args.serve:
        root = Path(args.serve)
        while True:
            request = root / "request.json"
            if not request.exists():
                time.sleep(.01)
                continue
            data = json.loads(request.read_text())
            request.unlink()
            t0 = time.perf_counter()
            encode((root / "pixels.rgb").read_bytes(), root / "rows.bf16", data["count"])
            phase["request_seconds"] = time.perf_counter() - t0
            phase["pid"] = os.getpid()
            marker = root / "done.tmp"
            marker.write_text(json.dumps(phase))
            marker.replace(root / "done.json")
    else:
        encode(raw, target, count)
    phase["total_seconds"] = time.perf_counter() - started
    print("ANE_WORKER_TIMING " + json.dumps(phase), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
