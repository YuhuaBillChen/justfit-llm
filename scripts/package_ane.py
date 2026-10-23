"""Package a qualified local ANE tier and its compiled cache for matching devices."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import re
import stat
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from justfit_llm.ane_bundle import worker_files, MAX_ARCHIVE
from justfit_llm.runtime_bundle import sha


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--assets-root',type=Path,required=True);p.add_argument('--capacity',type=int,choices=(4096,8192),required=True)
 p.add_argument('--node',choices=('m4','m5','m6'),required=True);p.add_argument('--model',type=Path,required=True)
 p.add_argument('--qualification',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if platform.system()!='Darwin' or platform.machine()!='arm64':p.error('package ANE assets on the reference Mac')
 if a.output.exists():p.error('refusing to overwrite an archive')
 proof=json.loads(a.qualification.read_text());tier=a.assets_root/str(a.capacity);catalog=json.loads((tier/'catalog.json').read_text())
 assert proof['passed'] and proof['node']==a.node and proof['worker_files']==worker_files(ROOT)
 weights_sha=sha(a.model/'vision/model.bin');assert proof['vision_weights_sha256']==weights_sha
 source=tier/'assets';files={};groups=[];cache=Path.home()/'Library/Application Support/coreai-cache'
 for i,row in enumerate(catalog['groups']):
  asset=source/f'group-{i:02d}'/'model.aimodel'
  assert row['asset_hash']==(asset/'main.hash').read_bytes().hex()
  compiled=Path(row['guard']['path']).parents[3]
  groups.append(dict(asset_hash=row['asset_hash'],cache_relative=compiled.relative_to(cache).as_posix()))
  for prefix,folder in [(f'assets/group-{i:02d}/model.aimodel',asset),(f'compiled/group-{i:02d}',compiled)]:
   for file in sorted(folder.rglob('*')):
    if file.is_file():
     assert not file.is_symlink(),file
     files[prefix+'/'+file.relative_to(folder).as_posix()]=file
 assert len(groups)==9
 files['positions.npy']=tier/'positions.npy';files['LICENSE']=ROOT/'LICENSE'
 notice=('JustFit precompiled ANE vision assets.\nDerived from the Apache-2.0 Qwen3.8 vision weights (Alibaba / mlx-community),\npacked by Inco AI at incoai/Qwen3.8-27B-Splash revision\n9d27070b71f7142c6b6025f03ac011d70a73cb48.\nJustFit graph conversion, conditioning and compiled-cache packaging are modifications.\nThe source model and its components retain their upstream Apache-2.0 license.\nSource: https://huggingface.co/incoai/Qwen3.8-27B-Splash\nEngine acknowledgments: https://github.com/incoai/splash\nNo upstream endorsement is implied.\n').encode()
 refs=proof['tiers'][str(a.capacity)]
 meta=dict(schema=1,kind='justfit-ane-vision',node=a.node,capacity=a.capacity,architecture='arm64',
  chip=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
  os_build=subprocess.check_output(['sw_vers','-buildVersion'],text=True).strip(),
  vision_weights_sha256=weights_sha,worker_files=worker_files(ROOT,include_binaries=False),
  helper_abi=1,helper_build_contract_sha256=sha(ROOT/'build_release.py'),
  reference_native_helpers={n:h for n,h in proof['worker_files'].items() if n.endswith('.dylib')},groups=groups,
  reference_qualification_sha256=sha(a.qualification),
  reference_qualification=dict(passed=True,node=a.node,worker_upper_bytes=refs['worker_upper_bytes'],phase_qualification=refs['phase_qualification']),
  scope='Matching chip and OS build only; precompiled graph/cache files with reference-device qualification, not a universal CoreAI cache.',
  files={n:sha(f) for n,f in files.items()})
 meta['files']['NOTICE']=hashlib.sha256(notice).hexdigest()
 # Scan the actual payload, including opaque graph/compiler files, without loading a tier into RAM.
 for name,file in files.items():
  tail=b''
  with file.open('rb') as stream:
   for block in iter(lambda:stream.read(1024*1024),b''):
    data=tail+block
    if re.search(rb'/(?:Users|home)/[A-Za-z0-9_.-]+/|\b(?:mac|mbp|linux)-[0-9]+\b',data):raise ValueError('private identifier in '+name)
    tail=data[-256:]
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(a.output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  def info(name):
   i=zipfile.ZipInfo(name,date_time=(2026,1,1,0,0,0));i.create_system=3;i.external_attr=(stat.S_IFREG|0o600)<<16;i.compress_type=zipfile.ZIP_DEFLATED;return i
  z.writestr(info('bundle.json'),(json.dumps(meta,indent=2)+'\n').encode());z.writestr(info('NOTICE'),notice)
  for name,file in sorted(files.items()):
   with file.open('rb') as src,z.open(info(name),'w',force_zip64=True) as dst:
    for block in iter(lambda:src.read(1024*1024),b''):dst.write(block)
 if a.output.stat().st_size>MAX_ARCHIVE:raise ValueError('ANE archive exceeds the per-asset size limit')
 print(json.dumps(dict(file=a.output.name,bytes=a.output.stat().st_size,sha256=sha(a.output),metadata=meta),indent=2))
if __name__=='__main__':main()
