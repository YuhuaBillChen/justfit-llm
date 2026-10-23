"""Cached-only grouped vision worker, streamed models and zero-copy group boundaries."""
import argparse,gc,json,os,sys,time,hashlib,subprocess
from pathlib import Path
import numpy as np
from warm_chain import cache_guard,B
from partition import group_count
from ane_placement import require_ane
SERVING=Path(__file__).resolve().parent
sys.path.insert(0,str(SERVING))
from vision_geometry import patchify_into,interpolated_positions,rope_tables
from variable_vision_worker import write_bf16_rows

def record_cache(assets,path):
 build=subprocess.check_output(['/usr/bin/sw_vers','-buildVersion'],text=True).strip();entries=[]
 for i in range(group_count(assets)):
  asset=Path(assets)/f'group-{i:02d}'/'model.aimodel';guard=cache_guard(asset);root=Path(guard['path']).parents[3]
  files=[]
  for f in sorted(root.rglob('*')):
   if f.is_file():
    stat=f.stat();files.append([str(f),stat.st_size,stat.st_mtime_ns])
  if not files:raise ValueError('empty specialization')
  entries.append(dict(asset_hash=(asset/'main.hash').read_bytes().hex(),guard=guard,files=files))
 with Path(path).open('x') as out:json.dump(dict(build=build,groups=entries,partition_sha256=hashlib.sha256((Path(assets)/'partition.json').read_bytes()).hexdigest() if group_count(assets)==27 else None),out,indent=2)

def check_catalog(assets,path):
 record=json.loads(Path(path).read_text());build=subprocess.check_output(['/usr/bin/sw_vers','-buildVersion'],text=True).strip()
 if record['build']!=build or len(record['groups'])!=group_count(assets):raise ValueError('offline prewarm required: OS/group count changed')
 if group_count(assets)==27 and record.get('partition_sha256')!=hashlib.sha256((Path(assets)/'partition.json').read_bytes()).hexdigest():raise ValueError('offline prewarm required: partition changed')
 for i,row in enumerate(record['groups']):check_cached_group(Path(assets)/f'group-{i:02d}'/'model.aimodel',row)
 return record

def check_cached_group(asset,row):
 if row['asset_hash']!=(Path(asset)/'main.hash').read_bytes().hex():raise ValueError('offline prewarm required: asset changed')
 if not row['files']:raise ValueError('empty cache record')
 for name,size,modified in row['files']:
  st=Path(name).stat()
  if st.st_size!=size or st.st_mtime_ns!=modified:raise ValueError('offline prewarm required: cache changed')
 if hashlib.sha256(Path(row['guard']['path']).read_bytes()).hexdigest()!=row['guard']['manifest_sha256']:raise ValueError('placement metadata changed')

class GroupedEncoder:
 def __init__(self,assets,positions,capacity,catalog):
  if capacity not in (4096,8192,12288,14400,16384):raise ValueError('unsupported grouped capacity')
  self.assets=Path(assets);self.capacity=capacity;self.catalog=Path(catalog);self.group_count=group_count(assets)
  if self.group_count==27 and capacity not in (12288,14400):raise ValueError('single-layer partition requires12K capacity')
  self.cache_record=check_catalog(self.assets,catalog)
  self.guards=[cache_guard(self.assets/f'group-{i:02d}'/'model.aimodel') for i in range(self.group_count)]
  self.table=np.load(positions,allow_pickle=False)
  if self.table.shape!=(2304,1152) or self.table.dtype!=np.float32 or not np.isfinite(self.table).all():raise ValueError('invalid position table')
 def encode(self,raw,target,grids):
  cache_record=check_catalog(self.assets,self.catalog)
  if not isinstance(grids,list) or not 1<=len(grids)<=4:raise ValueError('invalid grids')
  edge=160 if self.capacity in (14400,16384) else 128
  for grid in grids:
   if not isinstance(grid,list) or len(grid)!=2 or any(type(s) is not int or s<2 or s>edge or s%2 for s in grid) or grid[0]*grid[1]>self.capacity:raise ValueError('grid exceeds compiled capacity or2048edge')
  if len(raw)!=sum(h*w*768 for h,w in grids):raise ValueError('RGB payload length mismatch')
  metrics=[];offset=0
  with Path(target).open('xb') as dst:
   for h,w in grids:
    count=h*w;pixels=np.frombuffer(raw,np.uint8,count*768,offset).reshape(h*16,w*16,3);offset+=count*768
    side={};previous=None;buffers={};rows=[];start=time.monotonic()
    for i in range(self.group_count):
     asset=self.assets/f'group-{i:02d}'/'model.aimodel';check_cached_group(asset,cache_record['groups'][i]);tick=time.monotonic();model=B.Model(asset,compute='ane');fn=model.function('group');placement=require_ane(asset,'group');load=time.monotonic()-tick;ins={}
     for n in fn.input_names:
      spec=fn.spec('input',n)
      if n=='x':
       if previous is not None:
        if tuple(spec['shape'])!=previous.shape or spec['dtype']!=previous.dtype.name or tuple(spec['strides'])!=previous.strides:raise ValueError('group handoff layout mismatch')
        ins[n]=previous
       else:
        buf=fn.buffer('input',n)
        if buf.shape!=(self.capacity,1536):raise ValueError('patch input signature mismatch')
        buf.zero();patchify_into(pixels,buf.np[:count]);ins[n]=buf
      elif n in side:
       if side[n][0]!=spec:raise ValueError('side input layout mismatch')
       ins[n]=side[n][1]
      else:
       buf=fn.buffer('input',n)
       if n=='position':
        buf.zero();interpolated_positions(self.table,h,w,48,out=buf.np[0,:,0,:count].T)
       elif n in ('cosine','sine'):
        cos,sin=rope_tables(h,w,72);buf.np[:]=1 if n=='cosine' else 0;buf.np[0,:,0,:count]=(cos if n=='cosine' else sin).T;del cos,sin
       elif n=='key_mask':buf.np[:]=-10000;buf.np[:,:count]=0
       else:raise ValueError('unknown graph input')
       ins[n]=buf;side[n]=(spec,buf)
     spec=fn.spec('output','output');key=(i%2,tuple(spec['shape']),spec['dtype'],tuple(spec['strides']))
     if key not in buffers:buffers[key]=fn.buffer('output','output')
     output=buffers[key];binding=fn.bind(ins,{'output':output});tick=time.monotonic();B.run([binding]);seconds=time.monotonic()-tick;previous=output
     rows.append(dict(group=i,load_seconds=load,encode_seconds=seconds,placement=placement));del binding,ins,fn,model;gc.collect()
    if previous.shape!=(self.capacity//4,5120):raise ValueError('embedding signature mismatch')
    write_bf16_rows(dst,previous.np[:count//4]);metrics.append(dict(grid=[h,w],wall_seconds=time.monotonic()-start,groups=rows))
    del previous,buffers,side,pixels;gc.collect()
  return dict(images=metrics,pid=os.getpid(),runtime_torch_imported='torch' in sys.modules,scope='cached-only streaming group models; preserved geometry and zero-copy boundaries')

def parse_arguments(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--positions',type=Path,required=True);p.add_argument('--capacity',type=int,required=True);p.add_argument('--source',type=Path);p.add_argument('--target',type=Path);p.add_argument('--grids');p.add_argument('--serve',type=Path);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--make-record',action='store_true');p.add_argument('--prewarm',action='store_true');p.add_argument('request',nargs='*');a=p.parse_args(argv)
 if a.request:
  if len(a.request)!=3 or a.source or a.target or a.serve or a.prewarm or a.make_record:p.error('expected source target image_count without other request mode')
  a.source=Path(a.request[0]);a.target=Path(a.request[1])
  try:a.count=int(a.request[2])
  except ValueError:p.error('image_count must be an integer')
  if a.count<1 or a.count>4:p.error('image_count must be between1 and4')
 else:a.count=None
 if sum(bool(v) for v in (a.make_record,a.prewarm,a.serve,a.source))!=1:p.error('select exactly one worker mode')
 return a

def main():
 a=parse_arguments()
 if a.make_record:record_cache(a.assets,a.catalog);return
 encoder=GroupedEncoder(a.assets,a.positions,a.capacity,a.catalog)
 if a.prewarm:
  # Offline compilation has already executed every graph. Startup checks
  # every cache before loading, streams cached models, and retains none.
  for i in range(encoder.group_count):
   asset=a.assets/f'group-{i:02d}'/'model.aimodel';check_cached_group(asset,encoder.cache_record['groups'][i]);model=B.Model(asset,compute='ane');fn=model.function('group');require_ane(asset,'group');del fn,model;gc.collect()
  print(json.dumps(dict(prewarm='cached_load_validation_only',capacity=a.capacity,runtime_torch_imported='torch' in sys.modules)),flush=True);return
 if a.serve:
  a.serve.mkdir(exist_ok=True);(a.serve/'ready.json').write_text(json.dumps(dict(pid=os.getpid(),capacity=a.capacity,cache_guards=encoder.guards)))
  while True:
   request=a.serve/'request.json'
   if not request.exists():time.sleep(.01);continue
   data=json.loads(request.read_text());request.unlink();target=a.serve/'rows.bf16'
   if target.exists():target.unlink()
   if type(data.get('count')) is not int or data['count']!=len(data['grids']):raise ValueError('image count/grid mismatch')
   result=encoder.encode((a.serve/'pixels.rgb').read_bytes(),target,data['grids']);marker=a.serve/'done.tmp';marker.write_text(json.dumps(result));marker.replace(a.serve/'done.json')
 else:
  if not a.source or not a.target or not a.grids:raise ValueError('source/target/grids required')
  grids=json.loads(a.grids)
  if a.count is not None and (not isinstance(grids,list) or a.count!=len(grids)):raise ValueError('image count/grid mismatch')
  result=encoder.encode(a.source.read_bytes(),a.target,grids);a.target.with_suffix('.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);print('ANE_WORKER_TIMING '+json.dumps(result),file=sys.stderr,flush=True)
if __name__=='__main__':main()
