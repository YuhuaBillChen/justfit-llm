"""Preflight cached compilation before admitting a live vision worker."""
import hashlib
import json
import subprocess
from pathlib import Path


def build_id():
    return subprocess.check_output(['sw_vers', '-buildVersion'], text=True).strip()


def save_record(path, asset, placement):
    root = Path(placement['specialization_root'])
    files = []
    for file in sorted(root.rglob('*')):
        if file.is_file():
            stat = file.stat()
            files.append([str(file), stat.st_size, stat.st_mtime_ns])
    if not files:
        raise ValueError('empty compiled specialization')
    data = {'build': build_id(), 'asset_hash': (asset/'main.hash').read_bytes().hex(),
            'manifest': placement['manifest_path'],
            'manifest_sha256': placement['manifest_sha256'], 'files': files}
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data))
    temporary.replace(path)


def check_record(path, asset):
    data = json.loads(path.read_text())
    if data['build'] != build_id() or data['asset_hash'] != (asset/'main.hash').read_bytes().hex():
        raise ValueError('compiled cache requires offline prewarm')
    if not data['files']:
        raise ValueError('empty compiled cache record')
    for name, size, modified in data['files']:
        stat = Path(name).stat()
        if stat.st_size != size or stat.st_mtime_ns != modified:
            raise ValueError('compiled cache changed; offline prewarm required')
    if hashlib.sha256(Path(data['manifest']).read_bytes()).hexdigest() != data['manifest_sha256']:
        raise ValueError('compiled placement metadata changed')
    return data
