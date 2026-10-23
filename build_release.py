"""Build source, both verify libraries and cached-vision helpers without research directories."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--metal-bin',type=Path,required=True);p.add_argument('--jobs',type=int,default=4);a=p.parse_args()
 metal=a.metal_bin/'metal';link=a.metal_bin/'metallib';src=ROOT/'source';tools=ROOT/'vision-tools'
 assert metal.is_file() and link.is_file()
 commands=[]
 def run(args,cwd=src):
  commands.append([str(v) for v in args]);subprocess.run(commands[-1],cwd=cwd,check=True)
 run(['make',f'-j{a.jobs}',f'METAL={metal}',f'METALLIB={link}','all'])
 wide=src/'build/verify16';wide.mkdir(exist_ok=True)
 airs=[]
 names=['gdn','draft','attention_qkv','paged_attention_store','tq4_store','tq4_attention','tq4_rotated','sampling']
 for name in names:
  source=src/'runtime/metal/kernels/decode'/(name+'.metal')
  out=wide/(source.parent.name+'-'+source.stem+'.air');airs.append(out)
  run([metal,'-std=metal4.0','-O3','-Wall','-Wextra','-Werror','-Iruntime','-mmacosx-version-min=26.4','-DJUSTFIT_TARGET_VERIFY_ROWS=16u','-c',source,'-o',out])
 run([link,*airs,'-o',src/'build/justfit-verify16.metallib'])
 run(['xcrun','clang','-O2','-dynamiclib',tools/'attributed_memory.c','-o',tools/'attributed_memory.dylib'])
 run(['xcrun','swiftc','-O','-parse-as-library','-emit-library',tools/'CoreAIBridge.swift','-o',tools/'libcoreai_bridge.dylib'])
 outputs=[src/'build/justfit-engine',src/'build/justfit.metallib',src/'build/justfit-verify16.metallib',tools/'attributed_memory.dylib',tools/'libcoreai_bridge.dylib']
 def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 (ROOT/'build-receipt.json').write_text(json.dumps(dict(commands=commands,outputs={str(p.relative_to(ROOT)):sha(p) for p in outputs},toolchain={str(p):sha(p) for p in [metal,link]},status='build_complete_runtime_unqualified'),indent=2))
if __name__=='__main__':main()
