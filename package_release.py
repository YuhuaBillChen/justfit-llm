"""Create a reproducible project archive and validate a fresh extraction."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
TOPS = ('source', 'vision-tools', 'service', 'tests', 'scripts', 'config', 'docs', 'paper', 'website', 'python')
ROOT_FILES = ('CITATION.cff', 'README.md', 'README.zh-CN.md', 'QUICKSTART.md', 'CHANGELOG.md', 'CONTRIBUTING.md',
              'SECURITY.md', 'LICENSE', 'NOTICE', '.gitignore', 'ANE_POLICY.md', 'ORIGIN.json',
              'RELEASE_STATUS.json', 'validate_package.py', 'build_release.py', 'package_release.py',
              'pyproject.toml', 'setup.py', 'MANIFEST.in', 'requirements-runtime.txt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        parser.error('output must be outside the project tree')
    if output.exists():
        parser.error('refusing to overwrite an existing archive')
    files = {}
    for top in TOPS:
        for path in (ROOT / top).rglob('*'):
            rel = path.relative_to(ROOT)
            if (path.is_file() and not path.is_symlink()
                    and not any(x in ('build', '.venv', 'venv', '__pycache__', '.git')
                                or x.endswith(('.egg-info', '.dist-info')) for x in rel.parts)
                    and path.suffix not in ('.dylib', '.pyc')):
                files[rel.as_posix()] = path.read_bytes()
    for name in ROOT_FILES:
        files[name] = (ROOT / name).read_bytes()
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())}
    files['package-manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo('JustFit-2.3/' + name, date_time=(2026, 10, 21, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            executable = name == 'source/justfit' or name.endswith('.sh')
            info.external_attr = (0o100755 if executable else 0o100644) << 16
            archive.writestr(info, data)
    with tempfile.TemporaryDirectory() as temporary:
        with zipfile.ZipFile(output) as archive:
            archive.extractall(temporary)
        root = Path(temporary) / 'JustFit-2.3'
        subprocess.run([sys.executable, str(root / 'validate_package.py')], check=True)
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(root / 'tests'), '-v'], check=True)
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(root / 'service'), '-p', 'test_*.py', '-v'], check=True)
    receipt = dict(archive=output.name, sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                   pinned_files=len(manifest), fresh_extraction_validation='passed',
                   source_tests='passed', service_tests='passed', publicly_released=json.loads((ROOT / 'RELEASE_STATUS.json').read_text())['publicly_released'],
                   scope='source archive integrity and local tests; not device acceptance or index publication')
    output.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    output.with_suffix('.sha256').write_text(receipt['sha256'] + '  ' + output.name + '\n')
    (ROOT / 'package-manifest.json').write_bytes(files['package-manifest.json'])
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
