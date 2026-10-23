"""Install pinned, device-specific ANE graphs and compiled caches without compilation."""
from pathlib import Path
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from .runtime_bundle import sha, digest, download

MAX_ARCHIVE = 2 * 1024**3
MAX_EXPANDED = 4 * 1024**3


def identity():
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise ValueError('precompiled ANE assets require macOS arm64')
    def read(*args):
        return subprocess.check_output(args, text=True).strip()
    return dict(chip=read('sysctl', '-n', 'machdep.cpu.brand_string'),
                os_build=read('sw_vers', '-buildVersion'))


def cache_namespace():
    """Resolve CoreAI's client directory for this Python process."""
    foundation = ctypes.CDLL('/System/Library/Frameworks/Foundation.framework/Foundation')
    objc = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    objc.objc_msgSend.restype = ctypes.c_void_p
    def send(receiver, selector):
        return objc.objc_msgSend(receiver, objc.sel_registerName(selector))
    pool = send(objc.objc_getClass(b'NSAutoreleasePool'), b'new')
    try:
        value = send(send(objc.objc_getClass(b'NSBundle'), b'mainBundle'), b'bundleIdentifier')
        if not value:
            value = send(send(objc.objc_getClass(b'NSProcessInfo'), b'processInfo'), b'processName')
        pointer = send(value, b'UTF8String') if value else None
        name = ctypes.string_at(pointer).decode('utf-8') if pointer else ''
        if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', name):
            raise ValueError('unsupported Python CoreAI cache namespace')
        return name
    finally:
        send(pool, b'drain')


def local_cache_relative(group):
    parts = Path(group['cache_relative']).parts
    return Path(parts[0], cache_namespace(), *parts[2:])


def worker_files(root, *, include_binaries=True):
    return {p.name: sha(p) for p in (root/'vision-tools').iterdir()
            if p.suffix in (('.py', '.dylib', '.swift', '.c') if include_binaries else ('.py', '.swift', '.c')) and p.is_file()}


def safe_parents(path):
    for p in (path, *path.parents):
        if p.is_symlink():
            raise ValueError('symlink in ANE installation destination')


def rename_exclusive(source, destination):
    """Publish a complete directory atomically, without replacing an existing one."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == 'darwin':
        function = libc.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        arguments = [os.fsencode(source), os.fsencode(destination), 4]  # RENAME_EXCL
    elif sys.platform.startswith('linux'):
        function = libc.renameat2
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                             ctypes.c_char_p, ctypes.c_uint]
        arguments = [-100, os.fsencode(source), -100, os.fsencode(destination), 1]
    else:
        raise ValueError('exclusive directory installation is unsupported on this OS')
    function.restype = ctypes.c_int
    if function(*arguments):
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(destination)
        raise OSError(error, os.strerror(error), str(destination))


def default_cache_root():
    return Path.home()/'Library/Application Support/coreai-cache'


def regular_sha(path):
    """Reject links and special files before hashing installed native input."""
    safe_parents(path)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError('non-regular ANE installation file')
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    except OSError as error:
        raise ValueError('missing or unsafe ANE installation file') from error


def file_set(root):
    safe_parents(root)
    if not root.is_dir():
        raise ValueError('missing ANE installation directory')
    result = {}
    for p in root.rglob('*'):
        safe_parents(p)
        if not p.is_dir():
            result[p.relative_to(root).as_posix()] = regular_sha(p)
    return result


def verify_installed(destination, meta, cache_root):
    """Check pinned content on reuse and before any cached native load.

    Never repair or overwrite an unexpected cache. Local processes remain trusted;
    this is not a sandbox against a concurrent writer with the same user identity.
    """
    expected_assets = {n.removeprefix('assets/'): h for n, h in meta['files'].items()
                       if n.startswith('assets/')}
    if file_set(destination/'assets') != expected_assets:
        raise ValueError('installed ANE assets differ; no cache was overwritten')
    for name, want in meta['files'].items():
        if not name.startswith(('assets/', 'compiled/')) and regular_sha(destination/name) != want:
            raise ValueError('installed ANE asset SHA256 mismatch')
    for index, group in enumerate(meta['groups']):
        prefix = f'compiled/group-{index:02d}/'
        expected = {n.removeprefix(prefix): h for n, h in meta['files'].items() if n.startswith(prefix)}
        if file_set(cache_root/local_cache_relative(group)) != expected:
            raise ValueError('installed CoreAI cache differs; no cache was overwritten')


def validate_metadata(meta, root, node, model):
    if not isinstance(meta, dict) or node not in ('m4','m5','m6'):
        raise ValueError('invalid ANE bundle metadata or node')
    if meta.get('architecture') != 'arm64':
        raise ValueError('ANE bundle architecture mismatch')
    if meta.get('schema') != 1 or meta.get('kind') != 'justfit-ane-vision':
        raise ValueError('unknown ANE bundle format')
    current = identity()
    if meta.get('node') != node or any(meta.get(k) != v for k,v in current.items()):
        raise ValueError('ANE bundle does not match this chip and macOS build')
    if meta.get('capacity') not in (4096, 8192):
        raise ValueError('unsupported ANE patch capacity')
    if meta.get('vision_weights_sha256') != sha(model/'vision/model.bin'):
        raise ValueError('ANE bundle does not match the vision weights')
    if (meta.get('worker_files') != worker_files(root, include_binaries=False)
            or meta.get('helper_abi') != 1
            or meta.get('helper_build_contract_sha256') != sha(root/'build_release.py')):
        raise ValueError('ANE bundle requires matching worker/helper source and ABI')
    files = meta.get('files')
    if not isinstance(files, dict) or not 1 <= len(files) <= 4096:
        raise ValueError('invalid ANE bundle file manifest')
    for name, value in files.items():
        if not isinstance(name, str):
            raise ValueError('invalid ANE manifest filename')
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or '\\' in name or p.as_posix() != name:
            raise ValueError('unsafe ANE bundle path')
        if name not in ('positions.npy', 'LICENSE', 'NOTICE') and not re.fullmatch(
                r'(assets|compiled)/group-0[0-8]/.+', name):
            raise ValueError('unexpected ANE bundle path')
        digest(value)
    groups = meta.get('groups')
    if not isinstance(groups, list) or len(groups) != 9:
        raise ValueError('ANE bundle must contain nine groups')
    for index, row in enumerate(groups):
        if not isinstance(row, dict) or not isinstance(row.get('cache_relative'), str):
            raise ValueError('invalid ANE group metadata')
        digest(row.get('asset_hash'))
        parts = Path(row.get('cache_relative', '')).parts
        if (len(parts) != 5 or parts[0] != current['os_build']
                or parts[1] not in ('org.python.python', 'python') or parts[2] != row['asset_hash']
                or not re.fullmatch('[0-9A-Fa-f]{64}', parts[3]) or parts[4] != 'model.aimodelx'):
            raise ValueError('unsafe or incompatible CoreAI cache namespace')
        for prefix in (f'assets/group-{index:02d}/model.aimodel/', f'compiled/group-{index:02d}/'):
            if not any(n.startswith(prefix) for n in files):
                raise ValueError('incomplete ANE group')
    digest(meta.get('reference_qualification_sha256'))
    proof = meta.get('reference_qualification', {})
    if not isinstance(proof, dict):
        raise ValueError('invalid ANE reference qualification')
    if (proof.get('passed') is not True or proof.get('node') != node
            or type(proof.get('worker_upper_bytes')) is not int
            or not 0 < proof['worker_upper_bytes'] <= 1127*2**20
            or proof.get('phase_qualification') != dict(prefill=node != 'm6', decode=True)):
        raise ValueError('missing device-class ANE reference qualification')
    return meta


def install(root, archive, expected_sha256, node, model, *, cache_root=None):
    if node not in ('m4', 'm5', 'm6'):
        raise ValueError('invalid ANE node')
    root, archive, model = Path(root).resolve(), Path(archive), Path(model).resolve()
    if archive.stat().st_size > MAX_ARCHIVE or sha(archive) != digest(expected_sha256):
        raise ValueError('ANE archive size or SHA256 mismatch')
    cache_root = Path(cache_root) if cache_root is not None else default_cache_root()
    asset_root = root/'local-state/ane'/node
    safe_parents(asset_root); safe_parents(cache_root)
    asset_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    with (cache_root/'.justfit-install.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with zipfile.ZipFile(archive) as z:
            entries = z.infolist()
            if (len(entries) > 4097 or len({i.filename for i in entries}) != len(entries)
                    or 'bundle.json' not in z.namelist() or z.getinfo('bundle.json').file_size > 512*1024
                    or any(not stat.S_ISREG(i.external_attr >> 16) for i in entries)
                    or sum(i.file_size for i in entries) > MAX_EXPANDED):
                raise ValueError('invalid ANE archive members or size')
            meta = validate_metadata(json.loads(z.read('bundle.json')), root, node, model)
            if {i.filename for i in entries} != {'bundle.json', *meta['files']}:
                raise ValueError('unexpected ANE archive member')
            destination = asset_root/str(meta['capacity'])
            safe_parents(destination)
            if destination.exists():
                regular_sha(destination/'bundle.json')
                receipt = json.loads((destination/'bundle.json').read_text())
                if receipt != meta:
                    raise ValueError('existing ANE assets differ; use a new workspace')
                verify_installed(destination, meta, cache_root)
                return meta
            with tempfile.TemporaryDirectory(prefix='.ane-install-', dir=asset_root) as temporary:
                temp = Path(temporary); stage = temp/'stage'; stage.mkdir()
                for name, want in meta['files'].items():
                    target = stage/name; target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(name) as source, target.open('xb') as output:
                        total = 0
                        for block in iter(lambda: source.read(1024*1024), b''):
                            total += len(block)
                            if total > z.getinfo(name).file_size:
                                raise ValueError('ANE member exceeds declared size')
                            output.write(block)
                    if sha(target) != want:
                        raise ValueError('ANE member SHA256 mismatch: '+name)
                    target.chmod(0o600)
                for i, group in enumerate(meta['groups']):
                    asset = stage/f'assets/group-{i:02d}/model.aimodel'
                    if (asset/'main.hash').read_bytes().hex() != group['asset_hash']:
                        raise ValueError('ANE graph identity mismatch')
                    source = stage/f'compiled/group-{i:02d}'
                    target = cache_root/local_cache_relative(group); safe_parents(target)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        if file_set(target) != file_set(source):
                            raise ValueError('existing CoreAI cache differs; no cache was overwritten')
                    else:
                        with tempfile.TemporaryDirectory(prefix='.justfit-cache-', dir=target.parent) as folder:
                            copied = Path(folder)/'model.aimodelx'; shutil.copytree(source, copied)
                            try: rename_exclusive(copied, target)
                            except FileExistsError:
                                if file_set(target) != file_set(source):
                                    raise ValueError('concurrent CoreAI cache differs')
                shutil.rmtree(stage/'compiled')  # Only our verified staging copy.
                (stage/'bundle.json').write_text(json.dumps(meta,indent=2)+'\n')
                rename_exclusive(stage, destination)
    return meta


def prepare_profile(root, node, model):
    """Validate cached loads, then bind reference evidence to local paths/pins."""
    root, model = Path(root).resolve(), Path(model).resolve()
    assets = root/'local-state/ane'/node
    if not all((assets/str(c)/'bundle.json').is_file() for c in (4096,8192)):
        return None
    tools = root/'vision-tools'; records = {}
    for capacity in (4096,8192):
        tier = assets/str(capacity)
        regular_sha(tier/'bundle.json')
        meta = json.loads((tier/'bundle.json').read_text())
        validate_metadata(meta, root, node, model)
        verify_installed(tier, meta, default_cache_root())
        catalog = tier/'catalog.json'
        if not catalog.exists():
            subprocess.run([sys.executable, str(tools/'provision.py'), 'catalog', '--root', str(assets), '--capacity', str(capacity)],check=True)
        subprocess.run([sys.executable, str(tools/'grouped_worker.py'), '--assets', str(tier/'assets'), '--positions', str(tier/'positions.npy'), '--capacity', str(capacity), '--catalog', str(catalog), '--prewarm'],check=True)
        records[str(capacity)] = dict(meta['reference_qualification'],
            asset_files=file_set(tier/'assets'), position_table_sha256=sha(tier/'positions.npy'),
            catalog_sha256=sha(catalog), reference_qualification_sha256=meta['reference_qualification_sha256'])
    proof = dict(passed=True,node=node,vision_weights_sha256=sha(model/'vision/model.bin'),
        worker_files=worker_files(root),tiers=records,
        scope='vendor reference qualification for this exact chip/OS/graphs/helpers, plus local full-ANE cached-load validation; not a new per-unit speed benchmark')
    evidence = assets/'installed-qualification.json'
    if not evidence.exists(): evidence.write_text(json.dumps(proof,indent=2)+'\n')
    elif json.loads(evidence.read_text()) != proof: raise ValueError('installed ANE evidence changed')
    profile = assets/'qualified/tiered.json'
    if not profile.exists():
        subprocess.run([sys.executable,str(tools/'provision.py'),'profile','--root',str(assets),'--model',str(model),'--node',node,'--worker-python',sys.executable,'--qualification',str(evidence)],check=True)
    return profile


def fetch(root, url, manifest_sha256, node, model):
    with tempfile.TemporaryDirectory(prefix='justfit-ane-download-') as folder:
        temp = Path(folder); manifest = temp/'index.json'
        download(url, manifest, 512*1024)
        if sha(manifest) != digest(manifest_sha256):
            raise ValueError('ANE download manifest SHA256 mismatch')
        data = json.loads(manifest.read_text()); current = identity()
        if data.get('schema') != 1 or data.get('kind') != 'justfit-ane-downloads':
            raise ValueError('unknown ANE download index')
        rows = [r for r in data.get('artifacts',[]) if r.get('node') == node and r.get('os_build') == current['os_build'] and r.get('chip') == current['chip']]
        if len(rows) != 2 or {r.get('capacity') for r in rows} != {4096,8192}:
            raise ValueError('no complete compatible 4K/8K download set')
        for row in rows:
            archive = temp/(str(row['capacity'])+'.zip')
            download(row.get('url'), archive, MAX_ARCHIVE)
            meta = install(root, archive, row.get('sha256'), node, model)
            if meta['capacity'] != row['capacity']: raise ValueError('ANE download tier mismatch')
    return prepare_profile(root, node, model)
