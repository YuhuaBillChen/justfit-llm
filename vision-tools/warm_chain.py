"""Cached grouped ANE execution; never imports Torch on the serving path."""
import argparse,gc,json,resource,time,sys,subprocess,ctypes,threading,os
from pathlib import Path
import numpy as np
import coreai_bridge as B
from ane_placement import require_ane,validate_manifest

def cache_guard(asset):
 key=(asset/'main.hash').read_bytes().hex()
 build=subprocess.check_output(['/usr/bin/sw_vers','-buildVersion'],text=True).strip()
 root=Path.home()/'Library/Application Support/coreai-cache'/build
 matches=list(root.glob(f'*/{key}/*/model.aimodelx/main-this-delegates/MPSGraph/*/manifest.plist'))
 if not matches:
  raise RuntimeError(f'cached specialization absent: {asset}; offline prewarm required')
 accepted=[]
 for p in matches:
  try:accepted.append((p,validate_manifest(p.read_bytes(),'group')))
  except Exception:pass
 if len(accepted)!=1:raise RuntimeError(f'cached placement missing or ambiguous: {asset}, {len(accepted)}')
 return dict(path=str(accepted[0][0]),**accepted[0][1])

def rel(a,b):
 a=a.astype(np.float32);b=b.astype(np.float32)
 return float(np.linalg.norm(a-b)/np.linalg.norm(b))

def main():
 p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['stream','resident'],required=True);p.add_argument('--repeat',type=int,default=2);a=p.parse_args()
 a.out.mkdir(exist_ok=False)
 meter=ctypes.CDLL(str(Path(__file__).resolve().parent/'attributed_memory.dylib'));meter.attributed_pid.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64)];meter.attributed_self.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
 peaks=[0]*4;ledger_peaks=[0]*7;stop=threading.Event()
 def sample():
  with (a.out/'attributed-memory.jsonl').open('x') as log:
   while not stop.is_set():
    values=(ctypes.c_uint64*4)();ledger=(ctypes.c_uint64*7)();ok=bool(meter.attributed_pid(os.getpid(),values));ledger_ok=bool(meter.attributed_self(ledger))
    if ledger_ok:
     for j,value in enumerate(ledger):ledger_peaks[j]=max(ledger_peaks[j],value)
    if ok:
     for j,value in enumerate(values):peaks[j]=max(peaks[j],value)
    log.write(json.dumps(dict(time=time.time(),ok=ok,values=list(values),ledger_ok=ledger_ok,ledger=list(ledger)))+'\n');log.flush();stop.wait(.05)
 thread=threading.Thread(target=sample,daemon=True);thread.start()
 guards=[cache_guard(a.assets/f'group-{i:02d}'/'model.aimodel') for i in range(9)]
 fixture=np.load(a.fixture);count=int(fixture['count'])//4
 rows=[]; bindings=[];functions=[];previous=None;placements=[]
 # One set of read-only side inputs is shared by resident groups when layouts match.
 shared={};setup=time.monotonic()
 if a.mode=='resident':
  for i in range(9):
   asset=a.assets/f'group-{i:02d}'/'model.aimodel';fn=B.Model(asset,compute='ane').function('group');functions.append(fn);placements.append(require_ane(asset,'group'))
   ins={}
   for name in fn.input_names:
    spec=fn.spec('input',name)
    if name=='x' and previous is not None:
     assert tuple(spec['shape'])==previous.shape and spec['dtype']==previous.dtype.name
     assert tuple(spec['strides'])==previous.strides,'incompatible group boundary layout'
     ins[name]=previous
    elif name in shared and shared[name][0]==spec:ins[name]=shared[name][1]
    else:
     buf=fn.buffer('input',name);buf.np[:]=fixture['patches' if name=='x' else name];ins[name]=buf
     if name!='x':shared[name]=(spec,buf)
   outs={n:fn.buffer('output',n) for n in fn.output_names};previous=outs['output'];bindings.append(fn.bind(ins,outs))
  plan=B.Plan(bindings)
 setup_seconds=time.monotonic()-setup
 for repeat in range(a.repeat):
  start=time.monotonic();group_rows=[]
  if a.mode=='resident':
   times=plan.run(times=True);output=previous.np.copy();group_rows=[dict(group=i,encode_seconds=float(t)/1000) for i,t in enumerate(times)]
  else:
   x=fixture['patches']
   for i in range(9):
    tick=time.monotonic();asset=a.assets/f'group-{i:02d}'/'model.aimodel';model=B.Model(asset,compute='ane');fn=model.function('group');placement=require_ane(asset,'group');load=time.monotonic()-tick
    ins={n:fn.buffer('input',n) for n in fn.input_names};outs={n:fn.buffer('output',n) for n in fn.output_names}
    for name,buf in ins.items():buf.np[:]=x if name=='x' else fixture[name]
    binding=fn.bind(ins,outs);tick=time.monotonic();B.run([binding]);encode=time.monotonic()-tick;x=outs['output'].np.copy()
    group_rows.append(dict(group=i,load_seconds=load,encode_seconds=encode,placement=placement))
    del binding,ins,outs,fn,model;gc.collect()
   output=x
  assert np.isfinite(output).all()
  np.save(a.out/f'output-{repeat}.npy',output)
  row=dict(attributed_peaks_bytes=list(peaks),repeat=repeat,wall_seconds=time.monotonic()-start,group_rows=group_rows,peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,finite=True)
  reference=a.assets/'full-fp16.npy'
  if reference.exists() and a.fixture.resolve()==(a.assets/'fixture.npz').resolve():row['fp16_l2']=rel(output[:count],np.load(reference)[:count])
  if 'fp16' in fixture:row['fixture_fp16_l2']=rel(output[:count],fixture['fp16'][:count])
  if 'reference' in fixture:row['fixture_fp32_l2']=rel(output[:count],fixture['reference'][:count])
  rows.append(row);(a.out/'progress.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
 stop.set();thread.join(2)
 bitexact=np.array_equal(np.load(a.out/'output-0.npy'),np.load(a.out/f'output-{a.repeat-1}.npy'))
 (a.out/'summary.json').write_text(json.dumps(dict(attributed_peak_fields=['phys_footprint','neural_footprint','lifetime_max_neural_footprint','lifetime_max_phys_footprint'],attributed_peaks_bytes=peaks,ledger_peak_fields=['phys_footprint','graphics_footprint','graphics_nofootprint','neural_footprint','neural_nofootprint','neural_nofootprint_peak','neural_nofootprint_total'],ledger_peaks_bytes=ledger_peaks,worker_1127mib_phys_footprint_only=peaks[0]<=1127*2**20,mode=a.mode,cache_guards=guards,setup_seconds=setup_seconds,placements=placements,repeats=rows,repeat_bitexact=bitexact,memory_scope='process RSS and kernel attribution; phys-only check excludes separate neural accounting and is not a budget qualification'),indent=2))
if __name__=='__main__':main()
