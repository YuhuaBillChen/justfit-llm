"""Test actual distributions in new environments without touching production."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import venv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--sdist', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    wheel, sdist, output = args.wheel.resolve(), args.sdist.resolve(), args.output.absolute()
    assert wheel.is_file() and sdist.is_file()
    output.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
    results = []

    def run(command, cwd=output, expected=0):
        number = len(results)
        process = subprocess.run([str(x) for x in command], cwd=cwd, env=env,
                                 capture_output=True, text=True, timeout=600)
        log = output / f'{number:02d}.log'
        log.write_text(process.stdout + process.stderr)
        results.append(dict(command=[str(x) for x in command], returncode=process.returncode,
                            log=log.name, expected=expected))
        (output / 'commands.json').write_text(json.dumps(results, indent=2) + '\n')
        if process.returncode != expected:
            raise RuntimeError(f'{command}: exit{process.returncode}; see {log}')
        return process.stdout

    for name, artifact in (('wheel', wheel), ('sdist', sdist)):
        environment = output / (name + '-venv')
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / 'bin/python'
        cli = environment / 'bin/justfit-llm'
        run([python, '-m', 'pip', 'install', artifact])
        run([python, '-m', 'pip', 'check'])
        version = run([cli, '--version']).strip()
        assert version == '2.3.0', version
        assert json.loads(run([cli, 'verify']))['passed']
        workspace = output / (name + ' workspace')
        run([cli, 'init', workspace])
        run([cli, 'verify', '--project', workspace])
        run([python, '-m', 'server.server', '--help'], cwd=workspace / 'source')
        for node in ('m4', 'm5', 'm6'):
            plan = json.loads(run([cli, 'serve', '--project', workspace, '--node', node,
                                   '--model', '/missing/model with spaces', '--port', '12345', '--dry-run']))
            assert Path(plan['command'][0]).samefile(python)
            assert Path(plan['cwd']).samefile(workspace / 'source')
            assert ('--disable-ane' in plan['command']) == (node != 'm6')
        marker = workspace / 'user-owned-data.txt'
        marker.write_text('must survive package lifecycle')
        run([python, '-m', 'pip', 'uninstall', '-y', 'justfit-llm'])
        run([python, '-c', "import importlib.util; assert importlib.util.find_spec('justfit_llm') is None"])
        assert not cli.exists()
        assert marker.read_text() == 'must survive package lifecycle'
        if name == 'wheel':
            # An isolated file:// simple index tests package/version resolution
            # without publishing or mixing third-party dependency indexes.
            index = output / 'simple/justfit-llm'
            index.mkdir(parents=True)
            digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            (index / 'index.html').write_text(f'<a href="{wheel.as_uri()}#sha256={digest}">{wheel.name}</a>')
            run([python, '-m', 'pip', 'install', '--no-deps', '--no-cache-dir',
                 '--index-url', (output / 'simple').as_uri(), 'justfit-llm==2.3.0'])
        else:
            run([python, '-m', 'pip', 'install', '--no-deps', artifact])
        assert marker.read_text() == 'must survive package lifecycle'
        run([cli, 'verify'])
        run([python, '-m', 'pip', 'check'])
    receipt = dict(passed=True, version='2.3.0',
                   artifacts={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (wheel, sdist)},
                   checks=len(results), scope='fresh wheel/sdist install, dependency resolution, integrity, frontend import/help, preset plans, uninstall/reinstall and local simple-index resolution',
                   mac_native_build_and_inference=False, testpypi_uploaded=False,
                   limitation='Linux installation and planning do not qualify Mac native execution or actual TestPyPI downloads; no previous public wheel exists for a cross-version upgrade test.')
    (output / 'RESULT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
