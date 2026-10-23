"""Bounded routing checks against an isolated candidate server, not a live cluster."""
import argparse
import base64
import concurrent.futures
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'vision-tools'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--node', choices=['m4', 'm5', 'm6'], required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--assets-root', type=Path, required=True)
    parser.add_argument('--meter-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from provision import sha
    from memory_audit import audit
    args.output.mkdir(parents=True, exist_ok=False)

    def call(path, body=None):
        request = urllib.request.Request(args.base_url.rstrip('/')+path,
            data=None if body is None else json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)

    def save(name, value):
        (args.output/(name+'.json')).write_text(json.dumps(value, indent=2)+'\n')

    def status():
        return call('/status')

    def text(index, long=False):
        prompt = ('Write a long numbered list of practical tips about '+('gardening' if index == 0 else 'cooking')+
                  '. Explain each tip. Continue for at least 1000 tips.')
        if long:
            prompt = f'Independent prefill probe {index}.\n' + ''.join(f'Item {i}: gardens need soil, sunlight and water.\n' for i in range(600)) + '\nSummarize the items.'
        return call('/v1/chat/completions', dict(model='incoai/Qwen3.8-27B-Splash',
            messages=[dict(role='user', content=prompt)], reasoning_effort='none', temperature=0,
            max_completion_tokens=1024 if not long else 64))

    fixtures = {}
    expected = {}
    fixture_root = ROOT/'tests/fixtures/ane'
    for case in json.loads((fixture_root/'cases.json').read_text()):
        capacity = case['capacity']
        data = (fixture_root/case['image']).read_bytes()
        (args.output/f'{capacity}.png').write_bytes(data)
        expected[capacity] = case['expected']
        fixtures[capacity] = dict(model='incoai/Qwen3.8-27B-Splash', temperature=0,
            reasoning_effort='none', max_completion_tokens=128, messages=[dict(role='user', content=[
                dict(type='image_url', image_url=dict(url='data:image/png;base64,'+base64.b64encode(data).decode())),
                *case['content']])])

    def image(capacity):
        answer = call('/v1/chat/completions', fixtures[capacity])
        content = answer['choices'][0]['message']['content'].strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()
        assert all(json.loads(content).get(k) == v for k, v in expected[capacity].items()), answer
        return answer

    def count(s, capacity):
        return s['ane_vision'][str(capacity)]['encoded']

    initial = status()
    assert initial['requests']['submitted'] == initial['requests']['completed'], 'server must be isolated and idle'
    before = initial
    for capacity in fixtures:
        save(f'idle-{capacity}', image(capacity))
    after = status()
    assert all(count(after, c) == count(before, c) for c in fixtures), 'idle image must stay on GPU'
    samples = []
    stop = threading.Event()

    def monitor():
        while not stop.wait(.1):
            try:
                s = status()
                samples.append(dict(time=time.time(), scheduler=s['scheduler'], ane_ffn=s['ane_ffn'],
                                    ane_vision=s['ane_vision']))
            except OSError:
                pass

    thread = threading.Thread(target=monitor)
    thread.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            tasks = [pool.submit(text, i) for i in range(2)]
            deadline = time.monotonic()+240
            while time.monotonic() < deadline:
                current = status()
                if current['scheduler']['decoding'] >= 2 and not current['scheduler']['prefilling']:
                    break
                assert not any(t.done() for t in tasks), 'decode probe ended too soon'
                time.sleep(.1)
            else:
                raise TimeoutError('two decoding lanes')
            save('decode-arrival', current)
            for capacity in fixtures:
                previous = status()
                save(f'decode-{capacity}', image(capacity))
                current = status()
                assert count(current, capacity) == count(previous, capacity)+1, 'decode did not execute ANE'
                groups = current['ane_vision'][str(capacity)]['last_timings']['worker']['images'][0]['groups']
                assert len(groups) == 9 and all(g['placement']['full_ane'] and g['placement']['no_gpu'] for g in groups)
            save('decode-text', [t.result() for t in tasks])
        for capacity in fixtures:
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                ffn_before = status()['ane_ffn']['split_commands']
                task = pool.submit(text, capacity, True)
                deadline = time.monotonic()+240
                while time.monotonic() < deadline:
                    current = status()
                    if current['scheduler']['prefilling'] and (args.node != 'm6' or current['ane_ffn']['split_commands'] > ffn_before):
                        break
                    assert not task.done(), 'prefill probe ended too soon'
                    time.sleep(.1)
                else:
                    raise TimeoutError('active text prefill')
                save(f'prefill-arrival-{capacity}', current)
                previous = current
                save(f'prefill-{capacity}', image(capacity))
                current = status()
                assert count(current, capacity)-count(previous, capacity) == (0 if args.node == 'm6' else 1)
                task.result()
    finally:
        stop.set()
        thread.join(10)
        save('samples', samples)
    final = status()
    save('final-status', final)
    assert final['metal']['healthy'] and final['requests']['failed'] == 0
    assert final['ane_vision']['shared_payload']['reserved_bytes'] == 0
    memory = audit(args.meter_dir)
    save('memory', memory)
    assert memory['budget_passed'], memory
    tiers = {}
    for capacity in fixtures:
        tier = args.assets_root/str(capacity)
        assets = tier/'assets'
        tiers[str(capacity)] = dict(worker_upper_bytes=memory['conservative_lifetime_peak_sum_bytes'],
            phase_qualification=dict(prefill=args.node != 'm6', decode=True),
            position_table_sha256=sha(tier/'positions.npy'), catalog_sha256=sha(tier/'catalog.json'),
            asset_files={str(p.relative_to(assets)): sha(p) for p in assets.rglob('*') if p.is_file()})
    save('qualification', dict(node=args.node, passed=True,
        vision_weights_sha256=sha(args.model/'vision/model.bin'), tiers=tiers,
        worker_files={p.name: sha(p) for p in (ROOT/'vision-tools').iterdir()
                      if p.suffix in ('.py', '.dylib', '.swift', '.c')},
        scope='bounded idle GPU, B2 decode ANE, text-prefill policy, placement and physical/neural memory; not quality equivalence or sustained performance'))
    print('Passed; qualification receipt: '+str(args.output/'qualification.json'))


if __name__ == '__main__':
    main()
