"""Install an explicitly pinned macOS runtime; untrusted archives never choose paths."""
from pathlib import Path
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import tempfile
import urllib.request
import zipfile

OUTPUTS = ('source/build/justfit-engine', 'source/build/justfit.metallib', 'source/build/justfit-verify16.metallib',
           'vision-tools/attributed_memory.dylib', 'vision-tools/libcoreai_bridge.dylib')
MAX_BYTES = 256 * 1024 * 1024

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def digest(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('runtime SHA256 must be 64 lowercase hexadecimal characters')
    return value

class HTTPSOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject a transport downgrade before urllib sends the next request."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.startswith('https://'):
            raise ValueError('runtime redirect must use HTTPS')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url, output, limit):
    if not isinstance(url, str) or not url.startswith('https://'):
        raise ValueError('runtime downloads require HTTPS')
    opener = urllib.request.build_opener(HTTPSOnlyRedirectHandler())
    with opener.open(url, timeout=60) as response, output.open('xb') as target:
        if not response.geturl().startswith('https://'):
            raise ValueError('runtime redirect must use HTTPS')
        length = 0
        while True:
            block = response.read(min(1024 * 1024, limit - length + 1))
            if not block:
                break
            length += len(block)
            if length > limit:
                raise ValueError('runtime download exceeds size limit')
            target.write(block)

def validate_metadata(meta, root, version):
    if meta.get('schema') != 1 or meta.get('version') != version:
        raise ValueError('runtime version/schema mismatch')
    if meta.get('source_manifest_sha256') != sha(root / 'package-manifest.json'):
        raise ValueError('runtime does not match this source package')
    if platform.system() != 'Darwin' or platform.machine() != 'arm64' or meta.get('architecture') != 'arm64':
        raise ValueError('runtime requires macOS arm64')
    floor = meta.get('minimum_macos')
    if not isinstance(floor, list) or len(floor) != 2 or not all(type(v) is int and v >= 0 for v in floor):
        raise ValueError('invalid runtime macOS floor')
    if tuple(int(v) for v in platform.mac_ver()[0].split('.')[:2]) < tuple(floor):
        raise ValueError('runtime requires a newer macOS release')
    records = meta.get('outputs')
    if not isinstance(records, dict) or set(records) != set(OUTPUTS):
        raise ValueError('runtime output list mismatch')
    for h in records.values():
        digest(h)

def install(root, archive, expected_sha256, version):
    root = Path(root).resolve(); archive = Path(archive)
    if sha(archive) != digest(expected_sha256):
        raise ValueError('runtime archive SHA256 mismatch')
    # Validate every member before writing any native output. Never extractall.
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        if len(entries) != 6 or {i.filename for i in entries} != {'runtime.json', *OUTPUTS}:
            raise ValueError('unexpected or duplicate runtime archive member')
        if any(not stat.S_ISREG(i.external_attr >> 16) for i in entries):
            raise ValueError('runtime archive must contain regular files only')
        if sum(i.file_size for i in entries) > MAX_BYTES or z.getinfo('runtime.json').file_size > 65536:
            raise ValueError('runtime archive exceeds size limit')
        meta = json.loads(z.read('runtime.json')); validate_metadata(meta, root, version)
        for name in OUTPUTS:
            path = root / name
            if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != root and p.is_relative_to(root)):
                raise ValueError('unsafe runtime destination')
        with tempfile.TemporaryDirectory(prefix='.runtime-', dir=root) as temp:
            temp = Path(temp)
            for index, name in enumerate(OUTPUTS):
                target = temp / str(index); total = 0
                with z.open(name) as source, target.open('xb') as dest:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        total += len(block)
                        if total > MAX_BYTES:
                            raise ValueError('runtime member exceeds size limit')
                        dest.write(block)
                if sha(target) != meta['outputs'][name]:
                    raise ValueError('runtime output SHA256 mismatch: ' + name)
                target.chmod(0o755 if name.endswith(('justfit-engine', '.dylib')) else 0o644)
            for index, name in enumerate(OUTPUTS):
                target = root / name; target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temp / str(index), target)
            receipt = dict(meta, status='prebuilt_installed_runtime_unqualified', archive_sha256=expected_sha256)
            (temp / 'receipt').write_text(json.dumps(receipt, indent=2) + '\n')
            os.replace(temp / 'receipt', root / 'build-receipt.json')
    return meta

def fetch_and_install(root, manifest_url, manifest_sha256, version):
    digest(manifest_sha256)
    with tempfile.TemporaryDirectory(prefix='justfit-runtime-') as temp:
        temp = Path(temp); manifest = temp / 'manifest.json'
        download(manifest_url, manifest, 65536)
        if sha(manifest) != manifest_sha256:
            raise ValueError('runtime download manifest SHA256 mismatch')
        meta = json.loads(manifest.read_text()); validate_metadata(meta, root, version)
        archive = temp / 'runtime.zip'
        download(meta.get('url'), archive, MAX_BYTES)
        return install(root, archive, digest(meta.get('archive_sha256')), version)
