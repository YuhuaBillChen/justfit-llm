"""Exercise the hard reasoning cap through the real HTTP/native path."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', default='incoai/Qwen3.8-27B-Splash')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []

    def save():
        (args.output/'result.json').write_text(json.dumps(dict(passed=False, cases=rows), indent=2)+'\n')

    def request(body):
        return urllib.request.Request(args.base_url.rstrip('/')+'/v1/chat/completions',
            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})

    def call(body):
        with urllib.request.urlopen(request(body), timeout=600) as response:
            return json.load(response)

    def body(budget, **options):
        return dict(model=args.model, temperature=0, seed=1, reasoning_effort='medium',
            chat_template_kwargs=dict(enable_thinking=True), thinking_budget=budget,
            max_completion_tokens=512, messages=[dict(role='user', content=
                'Compute the last six digits of 7 to the power 777. Think through repeated squaring step by step, checking every multiplication, before giving the final result.')], **options)

    def record(name, response, budget=None):
        usage = response['usage']
        count = usage['completion_tokens_details']['reasoning_tokens']
        message = response['choices'][0]['message']
        if budget is not None:
            assert count <= budget, (name, count, budget)
            assert message.get('content') or message.get('tool_calls'), (name, message)
        rows.append(dict(name=name, budget=budget, reasoning_tokens=count, response=response))
        save()
        return count

    save()
    control = body(0)
    control['max_completion_tokens'] = 128
    assert record('unbounded-positive-control', call(control)) > 31, 'fixture must genuinely need more than 31 reasoning tokens'
    for budget in (1, 8, 31, 32):
        count = record(f'cap-{budget}', call(body(budget)), budget)
        assert count == budget, 'the long-thinking fixture did not exercise the cap'
    production = body(1536)
    production['max_completion_tokens'] = 1792
    record('production-cap-1536', call(production), 1536)
    streamed = body(8)
    streamed.update(stream=True, stream_options=dict(include_usage=True))
    packets = []
    with urllib.request.urlopen(request(streamed), timeout=600) as response:
        for line in response:
            if not line.startswith(b'data:'):
                continue
            value = line[5:].strip()
            if value == b'[DONE]':
                break
            packet = json.loads(value)
            assert not packet.get('error'), packet
            packets.append(packet)
    usage = next(p['usage'] for p in reversed(packets) if p.get('usage'))
    assert usage['completion_tokens_details']['reasoning_tokens'] <= 8
    assert any(c.get('delta', {}).get('content') for p in packets for c in p.get('choices', []))
    (args.output/'stream.json').write_text(json.dumps(packets, indent=2)+'\n')
    rows.append(dict(name='stream-cap-8', reasoning_tokens=usage['completion_tokens_details']['reasoning_tokens']))
    save()
    structured = body(8)
    structured['messages'] = [dict(role='user', content='Think carefully about 17 times 19, then return the product as the answer field.')]
    structured['response_format'] = dict(type='json_schema', json_schema=dict(name='answer', strict=True,
        schema=dict(type='object', properties=dict(answer=dict(type='integer')), required=['answer'], additionalProperties=False)))
    answer = call(structured)
    record('structured-cap-8', answer, 8)
    assert json.loads(answer['choices'][0]['message']['content'])['answer'] == 323
    tool = body(8)
    tool['messages'] = [dict(role='user', content='Think carefully about 17 times 19, then call echo with the product as value.')]
    tool['tools'] = [dict(type='function', function=dict(name='echo', strict=True,
        parameters=dict(type='object', properties=dict(value=dict(type='integer')), required=['value'], additionalProperties=False)))]
    tool['tool_choice'] = dict(type='function', function=dict(name='echo'))
    answer = call(tool)
    record('tool-cap-8', answer, 8)
    calls = answer['choices'][0]['message']['tool_calls']
    assert calls[0]['function']['name'] == 'echo'
    assert json.loads(calls[0]['function']['arguments'])['value'] == 323
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        futures = []
        for budget in (1, 31):
            payload = body(budget)
            payload['messages'] = [dict(role='user', content='Think through how to organize a garden, then write a long numbered list of 200 practical gardening tips, explaining each tip. Continue until complete.')]
            futures.append((budget, pool.submit(call, payload)))
        for budget, future in futures:
            record(f'concurrent-cap-{budget}', future.result(), budget)
    for invalid in (True, -1, 1.5, '8'):
        try:
            call(body(invalid))
        except urllib.error.HTTPError as error:
            assert error.code == 400, error
        else:
            raise AssertionError(f'invalid budget accepted: {invalid!r}')
    for name, payload in [('disabled-thinking', body(8)), ('no-answer-room', body(511))]:
        if name == 'disabled-thinking':
            payload['reasoning_effort'] = 'none'
            payload['chat_template_kwargs']['enable_thinking'] = False
        try:
            call(payload)
        except urllib.error.HTTPError as error:
            assert error.code == 400, error
        else:
            raise AssertionError(name+' accepted')
    (args.output/'result.json').write_text(json.dumps(dict(passed=True, cases=rows,
        scope='hard token cap, speculative boundaries, cached-prefix reuse, stream, JSON, tools, concurrent per-request caps and HTTP validation'), indent=2)+'\n')
    print('Hard thinking budget checks passed: '+str(args.output/'result.json'))


if __name__ == '__main__':
    main()
