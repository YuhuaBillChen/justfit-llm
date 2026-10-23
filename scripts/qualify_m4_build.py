"""Finite, isolated old/new verify16 comparison; never changes a serving entry."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def call(port, path, body=None):
    request = urllib.request.Request(
        f'http://127.0.0.1:{port}{path}',
        data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--baseline-verify16', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vision-profile', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    preset = json.loads((ROOT / 'config/nodes/m4.json').read_text())
    prompts = [
        ('What is 17 multiplied by 19? Reply with the integer only.', '323'),
        ('Sort these integers ascending: 9, -2, 4, 0. Reply with a JSON array only.', '[-2, 0, 4, 9]'),
        ('Write JSON with key answer and value READY. No other text.', '{"answer":"READY"}'),
        ('In Python, what is len([2, 3, 5, 7, 11])? Reply with the integer only.', '5'),
        ('What is the capital of France? Answer with the city name only.', 'Paris'),
        ('Return the word BLUE exactly.', 'BLUE'),
    ]
    record = {'node': 'm4', 'scope': 'matched deterministic functional and bounded concurrency comparison; not a broad quality benchmark',
              'binary_sha256': sha(ROOT / 'source/build/justfit-engine'),
              'main_metallib_sha256': sha(ROOT / 'source/build/justfit.metallib'),
              'model_vision_sha256': sha(args.model / 'vision/model.bin'),
              'power': subprocess.check_output(['pmset', '-g', 'batt'], text=True), 'arms': []}
    assert "AC Power" in record['power'], 'M4 must be on AC'
    def save():
        (args.output / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    save()
    for name, library in [('baseline', args.baseline_verify16), ('candidate', ROOT / 'source/build/justfit-verify16.metallib')]:
        # Refuse overlapping native inference instead of stopping other processes.
        processes = subprocess.check_output(['ps', '-axo', 'command='], text=True)
        assert not any(' serve-native ' in line for line in processes.splitlines()), 'another native engine is running'
        state = args.output / name
        state.mkdir()
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        values = dict(release=str(ROOT), model=str(args.model), port=str(port), cache=str(state / 'cache'))
        command = [part.format(**values) for part in preset['command']]
        command[0] = str(args.python)
        environment = {k: v for k, v in os.environ.items()
                       if not k.startswith(('JUSTFIT_', 'DASH_', 'MLX_', 'APC_', 'ANE_'))
                       and k not in ('PYTHONPATH', 'PYTHONHOME')}
        environment.update(preset['environment'])
        environment.update(JUSTFIT_VERIFY16_LIBRARY=str(library), DASH_APC_DIR=str(state / 'cache'),
                           ANE_METER_DIR=str(state / 'ane-meter'))
        if args.vision_profile:
            environment['JUSTFIT_ANE_VISION_PROFILE'] = str(args.vision_profile)
        arm = {'name': name, 'verify16_sha256': sha(library), 'command': command,
               'environment': {k: environment[k] for k in preset['environment']}, 'cases': []}
        record['arms'].append(arm)
        with (state / 'server.log').open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT / 'source', env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + 180
                while time.monotonic() < deadline:
                    assert process.poll() is None, 'server exited'
                    try:
                        status = call(port, '/status')
                        if status.get('ready'):
                            break
                    except OSError:
                        pass
                    time.sleep(0.5)
                else:
                    raise TimeoutError('startup')
                def request(prompt, limit=128):
                    start = time.monotonic()
                    response = call(port, '/v1/chat/completions', dict(
                        model='incoai/Qwen3.8-27B-Splash', messages=[dict(role='user', content=prompt)],
                        reasoning_effort='none', temperature=0, max_completion_tokens=limit))
                    return dict(response=response, wall_seconds=time.monotonic() - start)
                for prompt, expected in prompts:
                    result = request(prompt)
                    answer = result['response']['choices'][0]['message']['content'].strip()
                    try:
                        correct = json.loads(answer) == json.loads(expected)
                    except (ValueError, TypeError):
                        correct = answer == expected
                    arm['cases'].append(dict(prompt=prompt, expected=expected, correct=correct, **result))
                    save()
                    assert correct, (name, prompt, answer)
                context = ''.join(f'Record {i}: the identifier is garden-{i:04d}.\n' for i in range(1200))
                long_prompt = context + '\nWhat is the identifier in record 1199? Reply with the identifier only.'
                arm['long_cold'] = request(long_prompt)
                arm['long_warm'] = request(long_prompt)
                assert 'garden-1199' in arm['long_warm']['response']['choices'][0]['message']['content']
                with concurrent.futures.ThreadPoolExecutor(2) as pool:
                    arm['concurrent'] = list(pool.map(lambda topic: request(
                        'Write a long numbered list of practical tips about ' + topic + '. Explain each tip.', 384),
                        ['gardening', 'cooking']))
                arm['final_status'] = call(port, '/status')
                assert arm['final_status']['metal']['healthy']
                assert arm['final_status']['requests']['failed'] == 0
                save()
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(10)
    old, new = record['arms']
    record['short_outputs_equal'] = all(
        a['response']['choices'][0]['message'] == b['response']['choices'][0]['message']
        for a, b in zip(old['cases'], new['cases']))
    record['long_outputs_equal'] = old['long_cold']['response']['choices'][0]['message'] == new['long_cold']['response']['choices'][0]['message']
    record['passed'] = record['short_outputs_equal'] and record['long_outputs_equal']
    save()
    assert record['passed'], 'matched outputs differ; inspect before qualification'
    print(json.dumps({'passed': True, 'result': str(args.output / 'result.json')}))


if __name__ == '__main__':
    main()
