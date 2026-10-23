"""Exercise reasoning policy at the final node broker, not just native HTTP."""
import argparse,concurrent.futures,json,time,urllib.request
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base-url',required=True);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args();W=args.output;W.mkdir(parents=True,exist_ok=False);rows=[]
def call(body):
 req=urllib.request.Request(args.base_url.rstrip('/')+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=900) as f:return json.load(f)
def body(cap,limit=512):return dict(model='qwen3.8-27b-mlx-vlm',reasoning=dict(max_tokens=cap),temperature=0,seed=1,max_completion_tokens=limit,messages=[dict(role='user',content='Compute the last six digits of 7 to the power 777. Think through repeated squaring step by step, checking every multiplication, before giving the final result.')])
def record(name,response,cap=None,exact=False):
 n=response['usage']['completion_tokens_details']['reasoning_tokens'];message=response['choices'][0]['message']
 if cap is not None:
  assert n<=cap,(name,n,cap)
  assert message.get('content') or message.get('tool_calls'),(name,message)
  if exact:assert n==cap,(name,n,cap)
 rows.append(dict(name=name,cap=cap,reasoning_tokens=n,response=response));(W/'budget-result.json').write_text(json.dumps(dict(passed=False,cases=rows),indent=2));return n
control=body(8,128);control.pop('reasoning');control['reasoning_effort']='max';assert record('unbounded-control',call(control))>32
for cap in (1,8,31,32):record(f'custom-cap-{cap}',call(body(cap)),cap,True)
b=body(1536,2048);b.pop('reasoning');b['model']='qwen3.8-27b-bounded-thinking';record('bounded-model-alias-1536',call(b),1536,True)
b=body(8);b.update(stream=True,stream_options=dict(include_usage=True));packets=[]
req=urllib.request.Request(args.base_url.rstrip('/')+'/v1/chat/completions',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=900) as f:
 for line in f:
  if not line.startswith(b'data:'):continue
  raw=line[5:].strip()
  if raw==b'[DONE]':break
  p=json.loads(raw);assert not p.get('error'),p;packets.append(p)
usage=next(p['usage'] for p in reversed(packets) if p.get('usage'));assert usage['completion_tokens_details']['reasoning_tokens']==8
assert any(c.get('delta',{}).get('content') for p in packets for c in p.get('choices',[]));(W/'budget-stream.json').write_text(json.dumps(packets,indent=2));rows.append(dict(name='stream-cap-8',reasoning_tokens=8))
b=body(8);b['messages']=[dict(role='user',content='Think carefully about 17 times 19, then call echo with the product as value.')];b['tools']=[dict(type='function',function=dict(name='echo',strict=True,parameters=dict(type='object',properties=dict(value=dict(type='integer')),required=['value'],additionalProperties=False)))];b['tool_choice']=dict(type='function',function=dict(name='echo'));a=call(b);record('tool-cap-8',a,8);assert json.loads(a['choices'][0]['message']['tool_calls'][0]['function']['arguments'])['value']==323
with concurrent.futures.ThreadPoolExecutor(2) as pool:
 tasks=[]
 for cap in (1,31):
  b=body(cap);b['messages']=[dict(role='user',content='Think how to organize a garden, then write 200 practical gardening tips. Explain each one and continue until complete.')];tasks.append((cap,pool.submit(call,b)))
 for cap,task in tasks:record(f'concurrent-cap-{cap}',task.result(),cap)
(W/'budget-result.json').write_text(json.dumps(dict(passed=True,cases=rows,scope='actual node broker custom budgets, bounded alias1536, stream, tools and concurrent per-request caps'),indent=2));print('Final broker budget passed',flush=True)
