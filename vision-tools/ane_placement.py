"""Fail closed unless the loaded specialization declares full ANE placement."""
import hashlib
import os
from pathlib import Path
import plistlib
import subprocess


class PlacementError(RuntimeError):
    pass


def validate_manifest(data, entrypoint):
    try:
        versions = plistlib.loads(data)['Package Version']
        matches = []
        for version in versions.values():
            for module in version['Optimized Modules'].values():
                for symbol, attributes in module['Entry Function Attributes'].items():
                    if symbol == entrypoint or symbol.startswith(entrypoint + '_'):
                        matches.append(attributes)
        if not matches or any(
            'mps.fullyPlacedOnANE' not in attrs or 'mps.noGPUActivity' not in attrs
            or not any(isinstance(a, dict) and a.get('mps.ane.calleeName') for a in attrs)
            for attrs in matches
        ):
            raise PlacementError('compiled function is not verified fully on ANE')
    except (KeyError, TypeError, ValueError, plistlib.InvalidFileException) as error:
        raise PlacementError('unknown compiled placement metadata') from error
    return {'full_ane': True, 'no_gpu': True, 'entries': len(matches),
            'manifest_sha256': hashlib.sha256(data).hexdigest()}


def require_ane(asset, entrypoint):
    key = (Path(asset) / 'main.hash').read_bytes().hex()
    try:
        result = subprocess.run(['/usr/sbin/lsof', '-nP', '-p', str(os.getpid()), '-Fn'],
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        raise PlacementError('cannot inspect loaded specialization') from error
    if result.returncode != 0:
        raise PlacementError('loaded specialization inspection failed')
    roots = set()
    for line in result.stdout.splitlines():
        if not line.startswith('n/'):
            continue
        path = Path(line[1:])
        if key not in path.parts or 'coreai-cache' not in path.parts:
            continue
        for candidate in (path, *path.parents):
            if candidate.name == 'model.aimodelx':
                roots.add(candidate)
                break
    if len(roots) != 1:
        raise PlacementError('loaded specialization is absent or ambiguous')
    root = roots.pop()
    manifests = list(root.glob('main-this-delegates/MPSGraph/*/manifest.plist'))
    if len(manifests) != 1:
        raise PlacementError('unsupported specialization layout')
    try:
        return dict(validate_manifest(manifests[0].read_bytes(), entrypoint),
                    manifest_path=str(manifests[0]), specialization_root=str(root))
    except OSError as error:
        raise PlacementError('placement metadata is unavailable') from error
