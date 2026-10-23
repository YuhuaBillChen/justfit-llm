"""Run a foreground, localhost JustFit server using a portable reference preset."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def plan(node, model, port, state, profile=None, python=None):
    preset = json.loads((ROOT / 'config/nodes' / (node + '.json')).read_text())
    model, state = Path(model).resolve(), Path(state).resolve()
    values = dict(release=str(ROOT), model=str(model), port=str(port), cache=str(state / 'cache'))
    command = [part.format(**values) for part in preset['command']]
    command[0] = str(python or ROOT / 'source/.venv/bin/python')
    environment = dict(preset['environment'])
    environment.update(DASH_APC_DIR=str(state / 'cache'), ANE_METER_DIR=str(state / 'ane-meter'),
                       JUSTFIT_VERIFY16_LIBRARY=str(ROOT / 'source/build/justfit-verify16.metallib'))
    if profile:
        environment['JUSTFIT_ANE_VISION_PROFILE'] = str(Path(profile).resolve())
    return dict(command=command, environment=environment, cwd=str(ROOT / 'source'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node', required=True, choices=('m4', 'm5', 'm6'))
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--state-dir', type=Path)
    parser.add_argument('--vision-profile', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--python', type=Path, help='use an already installed runtime environment')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    state = (args.state_dir or ROOT / 'local-state' / args.node).resolve()
    result = plan(args.node, args.model, args.port, state, args.vision_profile, args.python)
    if args.dry_run:
        print(json.dumps(result, indent=2))
        return
    if sys.platform != 'darwin':
        parser.error('actual inference requires an Apple silicon Mac')
    for path in (Path(result['command'][0]), ROOT / 'source/build/justfit-engine',
                 ROOT / 'source/build/justfit.metallib', ROOT / 'source/build/justfit-verify16.metallib'):
        if not path.is_file():
            parser.error('missing build or environment file: ' + str(path))
    for name in ('target', 'draft', 'tokenizer'):
        if not (args.model / name).is_dir():
            parser.error('model directory must contain ' + name + '/')
    if args.vision_profile and not args.vision_profile.is_file():
        parser.error('vision profile does not exist')
    for name in ('cache', 'ane-meter'):
        (state / name).mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('JUSTFIT_', 'DASH_', 'MLX_', 'APC_', 'ANE_'))
           and k not in ('PYTHONPATH', 'PYTHONHOME')}
    env.update(result['environment'])
    env['PYTHONNOUSERSITE'] = '1'
    os.chdir(result['cwd'])
    os.execve(result['command'][0], result['command'], env)


if __name__ == '__main__':
    main()
