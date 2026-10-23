"""Offline 4K/8K vision asset provisioning from a locally installed model pack."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

TOOLS = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def export_group(model, capacity, tier, index):
    import numpy as np
    import torch
    import coreai_torch
    from vision_graph import make_graph
    from vision_geometry import rope_tables
    target = tier / 'assets' / f'group-{index:02d}'
    if target.exists():
        raise FileExistsError(f'{target}: use a new output directory; existing assets are preserved')
    torch.set_num_threads(4)
    tower = make_graph(model, capacity=capacity, query_tile=1024).half()

    class SafeNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('weight', (tower.norm.weight.float() * (1152**.5 / 64)).half())
            self.register_buffer('bias', tower.norm.bias)

        def forward(self, x):
            center = x - x.mean(1, keepdim=True)
            small = center * (1 / 64)
            variance = (small * small).sum(1, keepdim=True)
            return center * torch.rsqrt(variance + 1e-6 * 1152 / 4096) * self.weight + self.bias

    tower.norm = SafeNorm()

    class Group(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList(tower.blocks[index*3:index*3+3])
            if index == 0:
                self.patch = tower.patch
            if index == 8:
                self.norm, self.merge1, self.merge2 = tower.norm, tower.merge1, tower.merge2

        def forward(self, x, position, cosine, sine, key_mask):
            if index == 0:
                x = self.patch(x.T[None, :, None, :]) + position
            for block in self.blocks:
                x = block(x, cosine, sine, key_mask)
            if index == 8:
                x = self.norm(x)[0, :, 0, :].T.reshape(capacity//4, 4608).T[None, :, None, :]
                h = self.merge1(x)
                x = self.merge2((h*.5) * (1 + torch.erf(h*.7071067811865475)))[0, :, 0, :].T
            return x

    group = Group().eval()
    del tower
    gc.collect()
    cosine, sine = rope_tables(64, capacity//64, 72)
    fixture = [np.zeros((1, 1152, 1, capacity), np.float16),
               cosine.T[None, :, None, :].astype(np.float16),
               sine.T[None, :, None, :].astype(np.float16),
               np.zeros((1, capacity, 1, 1), np.float16)]
    x = torch.zeros((capacity, 1536) if index == 0 else (1, 1152, 1, capacity), dtype=torch.float16)
    program = torch.export.export(group, (x, *[torch.from_numpy(v.copy()) for v in fixture]),
                                  strict=False).run_decompositions(coreai_torch.get_decomp_table())
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.add_exported_program(program, input_names=['x', 'position', 'cosine', 'sine', 'key_mask'],
                                   output_names=['output'], entrypoint_name='group')
    target.mkdir(parents=True, exist_ok=False)
    converter.to_coreai().save_asset(target / 'model.aimodel')
    from vision_weights import load_pack
    _, weights = load_pack(model)
    positions = tier / 'positions.npy'
    if not positions.exists():
        np.save(positions, weights['pos_embed'])
    receipt = dict(capacity=capacity, group=index, vision_weights_sha256=sha(model/'vision/model.bin'),
                   recipe='9x3-q8-merger-q64-pv1024-tanh-block-erf-merger',
                   files={str(p.relative_to(tier)): sha(p) for p in target.rglob('*') if p.is_file()},
                   tools={n: sha(TOOLS/n) for n in ('provision.py', 'vision_graph.py', 'vision_weights.py')})
    (target / 'export-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def compile_group(tier, index):
    # Explicit offline command: this is the only provisioning step that may compile.
    from coreai_bridge import Model
    from ane_placement import require_ane
    target = tier / 'assets' / f'group-{index:02d}' / 'model.aimodel'
    start = time.monotonic()
    model = Model(target, compute='ane')
    function = model.function('group')
    assert function
    row = dict(group=index, seconds=time.monotonic()-start, placement=require_ane(target, 'group'))
    (target.parent / 'compile-receipt.json').write_text(json.dumps(row, indent=2) + '\n')
    return row


def profile(model, root, node, python, evidence, *, probe=False):
    from grouped_worker import check_catalog
    if not probe:
        evidence = evidence.resolve(strict=True)
    destination = root / ('probe' if probe else 'qualified')
    if destination.exists():
        raise FileExistsError(f'{destination}: profiles are immutable; use a new directory')
    leaves = []
    documents = []
    for capacity in (4096, 8192):
        tier = root / str(capacity)
        assets, positions, catalog = tier/'assets', tier/'positions.npy', tier/'catalog.json'
        check_catalog(assets, catalog)
        # Identity hashes alone do not grant qualification. The explicit receipt
        # supplied by the operator must cover this device, placement and phases.
        phases = dict(prefill=node != 'm6', decode=True)
        if not probe:
            proof = json.loads(evidence.read_text())
            if proof.get('node') != node or proof.get('passed') is not True:
                raise ValueError('qualification receipt must pass and name the selected node')
            if proof.get('vision_weights_sha256') != sha(model/'vision/model.bin'):
                raise ValueError('qualification weight identity mismatch')
            bound = proof['tiers'][str(capacity)]['worker_upper_bytes']
            if type(bound) is not int or not 0 < bound <= 1127*2**20:
                raise ValueError('worker physical plus neural bound exceeds 1127 MiB')
            if proof['tiers'][str(capacity)]['phase_qualification'] != phases:
                raise ValueError('phase qualification does not match the node policy')
            if (proof['tiers'][str(capacity)].get('position_table_sha256') != sha(positions)
                    or proof['tiers'][str(capacity)].get('catalog_sha256') != sha(catalog)):
                raise ValueError('qualification position table or catalog mismatch')
            worker_files = {p.name: sha(p) for p in TOOLS.iterdir()
                            if p.suffix in ('.py', '.dylib', '.swift', '.c')}
            if proof.get('worker_files') != worker_files:
                raise ValueError('qualification worker/helper identity mismatch')
        files = [p for p in assets.rglob('*') if p.is_file()]
        files += [positions, catalog] + ([] if probe else [evidence])
        files += [p for p in TOOLS.iterdir() if p.suffix in ('.py', '.dylib', '.swift', '.c')]
        pins = {str(p.resolve()): sha(p) for p in files}
        if not probe and proof['tiers'][str(capacity)]['asset_files'] != {
                str(p.relative_to(assets)): sha(p) for p in assets.rglob('*') if p.is_file()}:
            raise ValueError('qualification asset pins mismatch')
        leaf = dict(schema=2, kind='grouped-vision', capacity_raw_patches=capacity, group_count=9,
                    worker_memory_bytes=1127*2**20, worker_timeout_seconds=600, startup_timeout_seconds=120,
                    vision_weights_sha256=sha(model/'vision/model.bin'), python=str(python.absolute()),
                    worker=str(TOOLS/'metered_worker.py'), assets=str(assets), position_table=str(positions),
                    cache_record=str(catalog), route_policy='gpu_relief', phase_qualification=phases,
                    qualification_status='unqualified_probe_only' if probe else 'operator_reviewed',
                    calibration_evidence=None if probe else str(evidence), files=pins)
        documents.append((f'{capacity}.json', leaf))
    # Validate every tier before publishing any profile files.
    destination.mkdir()
    for name, leaf in documents:
        path = destination / name
        path.write_text(json.dumps(leaf, indent=2) + '\n')
        leaves.append(dict(profile=path.name, sha256=sha(path)))
    with (destination/'tiered.json').open('x') as output:
        json.dump(dict(schema=3, kind='tiered-vision', tiers=leaves,
                       gpu_only_capacities=[12288, 14400, 16384]), output, indent=2)
    return dict(profile=str(destination/'tiered.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['export', 'compile', 'catalog', 'check', 'probe-profile', 'profile'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--capacity', type=int, choices=[4096, 8192])
    parser.add_argument('--group', type=int, choices=range(9))
    parser.add_argument('--node', choices=['m4', 'm5', 'm6'])
    parser.add_argument('--worker-python', type=Path)
    parser.add_argument('--qualification', type=Path)
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.model:
        args.model = args.model.resolve(strict=True)
    if args.action in ('profile', 'probe-profile'):
        if not all((args.model, args.node, args.worker_python)):
            parser.error('profile requires --model --node --worker-python')
        if args.action == 'profile' and not args.qualification:
            parser.error('qualified profile requires --qualification')
        result = profile(args.model, args.root, args.node, args.worker_python, args.qualification,
                         probe=args.action == 'probe-profile')
    else:
        if args.capacity is None:
            parser.error('--capacity is required')
        tier = args.root/str(args.capacity)
        if args.action == 'export':
            if args.model is None or args.group is None:
                parser.error('export requires --model and --group')
            result = export_group(args.model, args.capacity, tier, args.group)
        elif args.action == 'compile':
            if args.group is None:
                parser.error('compile requires --group')
            result = compile_group(tier, args.group)
        else:
            from grouped_worker import record_cache, check_catalog
            if args.action == 'catalog':
                record_cache(tier/'assets', tier/'catalog.json')
            else:
                check_catalog(tier/'assets', tier/'catalog.json')
            result = dict(action=args.action, capacity=args.capacity, passed=True)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
