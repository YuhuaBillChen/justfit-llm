"""Install, build and serve JustFit; `justfit serve` does all three on first run."""
import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import zipfile

from . import __version__


def payload_root():
    packaged = Path(__file__).resolve().parent / 'project'
    return packaged if packaged.is_dir() else Path(__file__).resolve().parents[2]


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'package-manifest.json').read_text())
    for name, digest in manifest.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe manifest path: ' + name)
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError('missing or unsafe source: ' + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('source digest mismatch: ' + name)
    return manifest


def initialize(destination):
    source = payload_root()
    manifest = verify(source)
    destination = Path(destination).absolute()
    # Exclusive directory creation also rejects dangling destination symlinks.
    destination.mkdir(parents=True, exist_ok=False)
    for name in [*manifest, 'package-manifest.json']:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
        if name == 'source/justfit' or name.endswith('.sh'):
            target.chmod(target.stat().st_mode | 0o111)
    verify(destination)
    return destination


DEFAULT_MODEL = 'incoai/Qwen3.8-27B-Splash'
# The runtime pack revision the reference presets were qualified with.
DEFAULT_REVISION = '9d27070b71f7142c6b6025f03ac011d70a73cb48'
DEFAULT_PORT = 8000
BUILD_OUTPUTS = ('source/build/justfit-engine', 'source/build/justfit.metallib', 'source/build/justfit-verify16.metallib',
                 'vision-tools/attributed_memory.dylib', 'vision-tools/libcoreai_bridge.dylib')
# Xcode 26+ installs the Metal toolchain as a downloadable component mounted here.
METAL_COMPONENT_GLOBS = (
    '/private/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-*/Metal.xctoolchain/usr/bin',
    '/Library/Developer/Toolchains/*.xctoolchain/usr/bin',
    '~/Library/Developer/Toolchains/*.xctoolchain/usr/bin',
    '/Applications/Xcode*.app/Contents/Developer/Toolchains/*.xctoolchain/usr/bin',
)
GIB = 1 << 30


def default_project():
    return Path.home() / 'Library/Application Support/JustFit' / __version__


def sysctl(name):
    return subprocess.run(['sysctl', '-n', name], capture_output=True, text=True, check=True).stdout.strip()


def select_preset(brand, memory_bytes):
    """Map an Apple chip name and memory size to the nearest qualified preset."""
    match = re.match(r'Apple M(\d+)\b', brand or '')
    if not match:
        raise ValueError(f'unrecognized chip {brand!r}; pass --node m4, m5 or m6')
    generation = int(match.group(1))
    if generation < 4:
        # The m4 preset's CPU prefill path needs SME, first present on M4.
        raise ValueError(f'{brand} is older than the tested M4 baseline; pass --node explicitly at your own risk')
    if generation > 6:
        raise ValueError(f'{brand} has no tested automatic preset; pass --node explicitly')
    if memory_bytes < 24 * GIB:
        raise ValueError(f'{memory_bytes / GIB:.0f} GiB memory is below the 24 GiB the presets require')
    return 'm4' if generation == 4 else 'm5' if generation == 5 else 'm6'


def detect_preset():
    return select_preset(sysctl('machdep.cpu.brand_string'), int(sysctl('hw.memsize')))


def find_metal_bin():
    """Return a directory containing both metal and metallib, or None."""
    try:
        found = [Path(subprocess.run(['xcrun', '-f', tool], capture_output=True, text=True, check=True)
                      .stdout.strip()).parent for tool in ('metal', 'metallib')]
        if found[0] == found[1]:
            return found[0]
    except (OSError, subprocess.CalledProcessError):
        pass
    for pattern in METAL_COMPONENT_GLOBS:
        for candidate in sorted(glob.glob(os.path.expanduser(pattern)), reverse=True):
            directory = Path(candidate)
            if (directory / 'metal').is_file() and (directory / 'metallib').is_file():
                return directory
    return None


def is_built(root):
    try:
        receipt = json.loads((root / 'build-receipt.json').read_text())
        if receipt.get('source_manifest_sha256') != hashlib.sha256((root / 'package-manifest.json').read_bytes()).hexdigest():
            return False
        outputs = receipt.get('outputs', {})
        return set(outputs) == set(BUILD_OUTPUTS) and all(
            not (root / name).is_symlink() and (root / name).is_file()
            and hashlib.sha256((root / name).read_bytes()).hexdigest() == outputs[name]
            for name in BUILD_OUTPUTS)
    except (OSError, ValueError, TypeError):
        return False


def build(root, metal_bin, jobs):
    verify(root)
    if jobs < 1:
        raise ValueError('--jobs must be positive')
    metal_bin = metal_bin or find_metal_bin()
    if metal_bin is None:
        raise ValueError('Metal toolchain not found; install it with '
                         '`xcodebuild -downloadComponent MetalToolchain` or pass --metal-bin')
    subprocess.run([sys.executable, str(root / 'build_release.py'),
                    '--metal-bin', str(Path(metal_bin).resolve()), '--jobs', str(jobs)], check=True)
    receipt_path = root / 'build-receipt.json'
    receipt = json.loads(receipt_path.read_text())
    receipt['source_manifest_sha256'] = hashlib.sha256((root / 'package-manifest.json').read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    if not is_built(root):
        raise ValueError('native build outputs do not match the build receipt')


def resolve_model(root, model, revision):
    """Use a local runtime pack directory, or download and verify a Hub repository."""
    path = Path(model).expanduser()
    if path.is_dir():
        return path.resolve()
    if model.startswith(('/', '.', '~')):
        raise ValueError(f'model directory does not exist: {model}')
    installer = [sys.executable, str(root / 'source/install/models.py'),
                 '--models', str(root / 'local-state/models'), '--model', model]
    if revision:
        installer += ['--revision', revision]
    subprocess.run(installer + ['prepare'], check=True)
    link = subprocess.run(installer + ['link'], capture_output=True, text=True, check=True).stdout.strip()
    return Path(link).resolve()


def require_mac():
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise ValueError('native build and inference require an Apple silicon Mac')
    version = tuple(int(x) for x in platform.mac_ver()[0].split('.')[:2])
    if version < (27, 0):
        raise ValueError('the complete JustFit runtime requires macOS 27.0 or later (CoreAI helpers)')


def add_ane_options(parser):
    options = parser.add_mutually_exclusive_group()
    options.add_argument('--ane-bundle', action='append', type=Path, default=[], help='local precompiled ANE tier ZIP; repeat for 4K and 8K')
    options.add_argument('--ane-manifest', help='HTTPS index of compatible precompiled 4K/8K ANE assets')
    parser.add_argument('--ane-sha256', action='append', default=[], help='matching SHA256 for each --ane-bundle')
    parser.add_argument('--ane-manifest-sha256', help='independently published SHA256 of the ANE download index')


def validate_ane_options(args):
    from .runtime_bundle import digest
    if len(args.ane_bundle) != len(args.ane_sha256):
        raise ValueError('supply one --ane-sha256 for each --ane-bundle')
    if bool(args.ane_manifest) != bool(args.ane_manifest_sha256):
        raise ValueError('ANE manifest and its SHA256 must be supplied together')
    if args.ane_manifest and not args.ane_manifest.startswith('https://'):
        raise ValueError('ANE downloads require HTTPS')
    for value in [*args.ane_sha256, *([args.ane_manifest_sha256] if args.ane_manifest_sha256 else [])]:
        digest(value)


def install_ane_assets(root, node, model, args):
    from . import ane_bundle
    if args.ane_manifest:
        return ane_bundle.fetch(root, args.ane_manifest, args.ane_manifest_sha256, node, model)
    for archive, expected in zip(args.ane_bundle, args.ane_sha256):
        ane_bundle.install(root, archive, expected, node, model)
    if args.ane_bundle:
        return ane_bundle.prepare_profile(root, node, model)
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='action', required=True)
    init = sub.add_parser('init', help='copy verified sources into a NEW writable workspace')
    init.add_argument('destination', type=Path)
    doctor = sub.add_parser('doctor', help='check platform, dependencies and optional Metal tools')
    doctor.add_argument('--metal-bin', type=Path)
    check = sub.add_parser('verify', help='verify bundled sources or an extracted workspace')
    check.add_argument('--project', type=Path)
    build_parser = sub.add_parser('build', help='build the native runtime and vision helpers on Mac')
    build_parser.add_argument('--project', type=Path, help='workspace (default: %(default)s)', default=default_project())
    build_parser.add_argument('--metal-bin', type=Path, help='directory with metal and metallib (default: auto-detect)')
    build_parser.add_argument('--jobs', type=int, default=4)
    serve = sub.add_parser('serve', help='run a foreground local server; sets up the workspace, build and model on first use')
    serve.add_argument('--project', type=Path, default=default_project(),
                       help='workspace, created on first use (default: %(default)s)')
    serve.add_argument('--node', choices=('m4', 'm5', 'm6'), help='preset (default: from chip and memory)')
    serve.add_argument('--model', default=DEFAULT_MODEL,
                       help='runtime pack directory or Hugging Face repository (default: %(default)s)')
    serve.add_argument('--revision', help='Hub revision (default: the qualified revision for the default model)')
    serve.add_argument('--port', type=int, default=DEFAULT_PORT)
    serve.add_argument('--metal-bin', type=Path, help='used only if a build is needed (default: auto-detect)')
    serve.add_argument('--jobs', type=int, default=4)
    runtime = serve.add_mutually_exclusive_group()
    runtime.add_argument('--runtime-bundle', type=Path, help='verified local prebuilt runtime ZIP')
    runtime.add_argument('--runtime-manifest', help='HTTPS runtime download manifest')
    serve.add_argument('--runtime-sha256', help='pinned SHA256 of the bundle or download manifest')
    serve.add_argument('--state-dir', type=Path)
    serve.add_argument('--vision-profile', type=Path)
    serve.add_argument('--no-ane-vision', action='store_true', help='use GPU vision even when precompiled ANE assets are installed')
    add_ane_options(serve)
    serve.add_argument('--dry-run', action='store_true')
    ane_parser = sub.add_parser('ane', help='install matching precompiled vision assets without an exporter or compiler')
    ane_actions = ane_parser.add_subparsers(dest='ane_action', required=True)
    ane_install = ane_actions.add_parser('install')
    ane_install.add_argument('--project', type=Path, default=default_project())
    ane_install.add_argument('--node', choices=('m4','m5','m6'))
    ane_install.add_argument('--model', default=DEFAULT_MODEL)
    ane_install.add_argument('--revision')
    add_ane_options(ane_install)
    args = parser.parse_args(argv)
    try:
        if args.action == 'init':
            print(initialize(args.destination))
        elif args.action == 'verify':
            print(json.dumps(dict(passed=True, files=len(verify(args.project or payload_root())))))
        elif args.action == 'doctor':
            try:
                require_mac(); supported = True; reason = None
            except ValueError as error:
                supported = False; reason = str(error)
            dependencies = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
            tools = {name: (args.metal_bin / name).is_file() for name in ('metal', 'metallib')} if args.metal_bin else {}
            print(json.dumps(dict(version=__version__, native_platform_supported=supported,
                                  reason=reason, dependencies_ok=dependencies.returncode == 0,
                                  dependency_details=dependencies.stdout + dependencies.stderr,
                                  metal_tools=tools), indent=2))
            if not supported or dependencies.returncode or (tools and not all(tools.values())):
                raise SystemExit(1)
        elif args.action == 'build':
            require_mac()
            build(args.project.resolve(), args.metal_bin, args.jobs)
        elif args.action == 'ane':
            validate_ane_options(args)
            if not (args.ane_bundle or args.ane_manifest):
                raise ValueError('supply a precompiled ANE bundle or download index')
            require_mac()
            root = args.project.resolve(); verify(root)
            if not is_built(root):
                raise ValueError('install or build the matching native runtime first')
            node = args.node or detect_preset()
            revision = args.revision or (DEFAULT_REVISION if args.model == DEFAULT_MODEL else None)
            model = resolve_model(root, args.model, revision)
            profile = install_ane_assets(root, node, model, args)
            print(json.dumps(dict(installed=True, vision_profile=str(profile) if profile else None,
                                  note='both 4K and 8K tiers are required for the automatic profile')))
        elif args.action == 'serve':
            validate_ane_options(args)
            if args.no_ane_vision and (args.vision_profile or args.ane_bundle or args.ane_manifest):
                raise ValueError('--no-ane-vision cannot be combined with ANE vision assets or a profile')
            if args.vision_profile and (args.ane_bundle or args.ane_manifest):
                raise ValueError('choose an existing vision profile or precompiled ANE assets')
            if not args.dry_run:
                require_mac()
            if not 1 <= args.port <= 65535:
                raise ValueError('--port must be between 1 and 65535')
            if args.jobs < 1:
                raise ValueError('--jobs must be positive')
            if bool(args.runtime_bundle or args.runtime_manifest) != bool(args.runtime_sha256):
                raise ValueError('runtime bundle/manifest and --runtime-sha256 must be supplied together')
            if args.runtime_sha256:
                from .runtime_bundle import digest
                digest(args.runtime_sha256)
            node = args.node or detect_preset()
            root = args.project.absolute()
            if not root.exists():
                if args.dry_run:
                    raise ValueError(f'workspace {root} does not exist; run without --dry-run to create it')
                print(f'Creating workspace {root}', file=sys.stderr)
                initialize(root)
            root = root.resolve()
            verify(root)
            if args.node is None:
                print(f'Using preset {node} for this Mac (override with --node)', file=sys.stderr)
            if not args.dry_run and (args.runtime_bundle or args.runtime_manifest):
                from .runtime_bundle import install, fetch_and_install
                if args.runtime_bundle:
                    install(root, args.runtime_bundle, args.runtime_sha256, __version__)
                else:
                    fetch_and_install(root, args.runtime_manifest, args.runtime_sha256, __version__)
                if not is_built(root):
                    raise ValueError('prebuilt runtime receipt verification failed')
            if not args.dry_run and not is_built(root):
                print('Building the native runtime (first run only)...', file=sys.stderr)
                build(root, args.metal_bin, args.jobs)
            revision = args.revision or (DEFAULT_REVISION if args.model == DEFAULT_MODEL else None)
            model = Path(args.model).expanduser() if args.dry_run else resolve_model(root, args.model, revision)
            prepared = None
            if not args.dry_run:
                prepared = install_ane_assets(root, node, model, args)
                if prepared is not None:
                    args.vision_profile = prepared
            automatic = root / 'local-state/ane' / node / 'qualified/tiered.json'
            if args.vision_profile is None and not args.no_ane_vision:
                if automatic.is_file():
                    args.vision_profile = automatic
            if (not args.dry_run and prepared is None and args.vision_profile is not None
                    and args.vision_profile.resolve() == automatic.resolve()):
                from . import ane_bundle
                if ane_bundle.prepare_profile(root, node, model) is None:
                    raise ValueError('automatic ANE profile has no complete installed bundles')
            command = [sys.executable, str(root / 'scripts/run_local.py'), '--python', sys.executable,
                       '--node', node, '--model', str(model.absolute()), '--port', str(args.port)]
            for flag, value in (('--state-dir', args.state_dir), ('--vision-profile', args.vision_profile)):
                if value is not None:
                    command.extend([flag, str(value.resolve())])
            if args.dry_run:
                command.append('--dry-run')
            subprocess.run(command, check=True)
    except (ValueError, OSError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
