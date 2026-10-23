"""Masked variable-grid ANE worker; compilation is confined to prewarm."""
import json
import os
from pathlib import Path
import resource
import sys
import time


def write_bf16_rows(output, rows):
    """Bound conversion scratch while preserving round-to-nearest-even bytes."""
    import numpy as np
    for start in range(0, len(rows), 32):
        values = rows[start:start + 32].astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError('non-finite ANE rows')
        bits = values.view(np.uint32)
        output.write(((bits + 0x7fff + ((bits >> 16) & 1)) >> 16)
                     .astype('<u2').tobytes())


def run(args):
    import numpy as np
    from vision_geometry import patchify_into, interpolated_positions, rope_tables
    from vision_cache_ready import check_record, save_record
    from ane_placement import require_ane
    asset = Path(args.asset)
    ready = Path(args.cache_record)
    if args.capacity != 4096 or not args.positions:
        raise ValueError('unsupported variable-grid profile')
    # A missing/changed cache fails before loading Core AI under LLM load.
    if not args.prewarm or args.cached_only:
        check_record(ready, asset)
    os.environ['COREAI_BRIDGE_LIB'] = str(Path(args.bridge)/'libcoreai_bridge.dylib')
    sys.path.insert(0, args.bridge)
    import coreai_bridge as bridge
    from ane_model_cache import load_function
    started = time.monotonic()
    model, function = load_function(bridge, asset, args.entrypoint,
                                    repair_cache=args.prewarm and not args.cached_only)
    placement = require_ane(asset, args.entrypoint)
    inputs = {name: function.buffer('input', name) for name in function.input_names}
    outputs = {name: function.buffer('output', name) for name in function.output_names}
    if inputs['patches'].np.shape != (4096, 1536) or outputs['embedding'].np.shape != (1024, 5120):
        raise ValueError('unexpected variable-grid signature')
    binding = function.bind(inputs, outputs)
    table = np.load(args.positions, allow_pickle=False)
    if table.shape != (2304, 1152) or table.dtype != np.float32 or not np.isfinite(table).all():
        raise ValueError('invalid position table')
    load_seconds = time.monotonic() - started

    def encode(raw, target, grids):
        if not isinstance(grids, list) or not 1 <= len(grids) <= 4:
            raise ValueError('invalid image count')
        for grid in grids:
            if (not isinstance(grid, (list, tuple)) or len(grid) != 2
                    or any(type(side) is not int or side < 2 or side % 2 for side in grid)
                    or grid[0] * grid[1] > args.capacity):
                raise ValueError('unsupported patch grid')
        if len(raw) != sum(h*w*768 for h,w in grids):
            raise ValueError('invalid RGB payload length')
        offset = 0
        metrics = []
        with target.open('wb') as output:
            for h, w in grids:
                count = h*w
                pixels = np.frombuffer(raw, np.uint8, count*768, offset).reshape(h*16,w*16,3)
                offset += count*768
                before = time.monotonic()
                inputs['patches'].np[:] = 0
                patchify_into(pixels, inputs['patches'].np[:count])
                inputs['position'].np[:] = 0
                interpolated_positions(table, h, w, 48,
                                       out=inputs['position'].np[0,:,0,:count].T)
                cosine, sine = rope_tables(h, w, 72)
                inputs['cosine'].np[:] = 1
                inputs['cosine'].np[0,:,0,:count] = cosine.T
                inputs['sine'].np[:] = 0
                inputs['sine'].np[0,:,0,:count] = sine.T
                del cosine, sine
                inputs['key_mask'].np[:] = -10000
                inputs['key_mask'].np[:,:count] = 0
                prepared = time.monotonic()-before
                before = time.monotonic()
                bridge.run([binding])
                elapsed = time.monotonic()-before
                write_bf16_rows(output, outputs['embedding'].np[:count//4])
                metrics.append({'grid':[h,w], 'patch_seconds':prepared, 'encode_seconds':elapsed})
        return {'images':metrics, 'read_and_load_seconds':load_seconds,
                'maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'placement':placement, 'pid':os.getpid()}

    if args.prewarm:
        import tempfile
        with tempfile.TemporaryDirectory(prefix='vision-warmup-') as name:
            encode(bytes(4*768), Path(name)/'rows.bf16', [[2,2]])
        save_record(ready, asset, placement)
        return
    if args.serve:
        folder = Path(args.serve)
        while True:
            request = folder/'request.json'
            if not request.exists():
                time.sleep(.01)
                continue
            data = json.loads(request.read_text())
            request.unlink()
            metrics = encode((folder/'pixels.rgb').read_bytes(), folder/'rows.bf16', data['grids'])
            marker = folder/'done.tmp'
            marker.write_text(json.dumps(metrics))
            marker.replace(folder/'done.json')
    else:
        metrics = encode(Path(args.source).read_bytes(), Path(args.target), json.loads(args.grids))
        print('ANE_WORKER_TIMING '+json.dumps(metrics), file=sys.stderr, flush=True)
