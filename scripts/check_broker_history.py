"""Verify synthetic long image history and warm reuse through a node broker."""
import argparse
import base64
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', default='qwen3.8-27b-mlx-vlm')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    image = base64.b64encode((ROOT/'tests/fixtures/ane/small.png').read_bytes()).decode()
    body = dict(model=args.model, reasoning_effort='none', temperature=0, max_completion_tokens=128,
        messages=[dict(role='system', content='Follow the user instructions and retain the conversation.'),
        dict(role='user', content='The secret for the original task is EMBER-73921-QX.\n'+'Garden soil water sunlight. '*2400),
        dict(role='assistant', content=None, tool_calls=[dict(id='call-original', type='function',
            function=dict(name='read_page', arguments='{"page":"original"}'))]),
        dict(role='tool', tool_call_id='call-original', content='Original task context retained.'),
        dict(role='user', content=[dict(type='image_url', image_url=dict(url='data:image/png;base64,'+image)),
            dict(type='text', text='Return only JSON with secret from the earliest user message, label and value for the largest completed jobs number in this screenshot. Use exactly keys secret, label, value.')])])
    answers = []
    for index in range(2):
        request = urllib.request.Request(args.base_url.rstrip('/')+'/v1/chat/completions',
            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=900) as response:
            answer = json.load(response)
        (args.output/f'response-{index}.json').write_text(json.dumps(answer, indent=2)+'\n')
        content = answer['choices'][0]['message']['content'].strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()
        assert json.loads(content) == dict(secret='EMBER-73921-QX', label='BETA', value=39)
        assert answer['usage']['prompt_tokens'] > 12288, answer['usage']
        answers.append(answer)
    usage = answers[1]['usage']
    cached = usage['prompt_tokens_details']['cached_tokens']
    assert cached >= usage['prompt_tokens']*.9, usage
    (args.output/'result.json').write_text(json.dumps(dict(passed=True,
        prompt_tokens=usage['prompt_tokens'], warm_cached_tokens=cached,
        scope='synthetic full-history retrieval, tool-pair preservation and screenshot answer; not real-workload performance'), indent=2)+'\n')
    print('Passed: '+str(args.output/'result.json'))


if __name__ == '__main__':
    main()
