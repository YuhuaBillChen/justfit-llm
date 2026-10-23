"""Create a relocatable runtime asset on a Mac after a verified local build.

Ad-hoc signing is for local qualification only. Public distribution should use
an explicitly selected Developer ID and its separate notarization workflow.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from justfit_llm import __version__
from justfit_llm.cli import is_built, verify
from justfit_llm.runtime_bundle import OUTPUTS, sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sign-identity', default='-', help='Developer ID identity; default ad-hoc for local testing only')
    a=p.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        p.error('create native runtime assets on macOS arm64')
    verify(ROOT)
    if not is_built(ROOT):
        p.error('build outputs or receipt do not match this package; run justfit build first')
    if a.output.exists():
        p.error('output already exists')
    with tempfile.TemporaryDirectory(prefix='justfit-package-runtime-') as temp:
        temp=Path(temp)
        for name in OUTPUTS:
            target=temp/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, target)
            if name.endswith('.dylib'):
                subprocess.run(['install_name_tool','-id','@rpath/'+target.name,str(target)], check=True)
                subprocess.run(['codesign','--force','--sign',a.sign_identity,str(target)], check=True)
            elif name.endswith('/justfit-engine') and a.sign_identity != '-':
                subprocess.run(['codesign','--force','--options','runtime','--timestamp','--sign',a.sign_identity,str(target)],check=True)
            if name.endswith(('.dylib','/justfit-engine')):
                subprocess.run(['codesign','--verify','--strict',str(target)],check=True)
                dependencies=subprocess.check_output(['otool','-L',str(target)],text=True).splitlines()[1:]
                for line in dependencies:
                    dependency=line.strip().split(' (',1)[0]
                    if not dependency.startswith(('/System/Library/','/usr/lib/','@rpath/'+target.name)):
                        raise ValueError('nonportable native dependency: '+dependency)
        meta=dict(schema=1,version=__version__,architecture='arm64',minimum_macos=[27,0],
                  source_manifest_sha256=sha(ROOT/'package-manifest.json'),
                  signing='ad-hoc-local-qualification' if a.sign_identity=='-' else 'developer-id-requested-notarization-separate',
                  outputs={name:sha(temp/name) for name in OUTPUTS})
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(a.output,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for name,data in [('runtime.json',(json.dumps(meta,indent=2)+'\n').encode()),*[(n,(temp/n).read_bytes()) for n in OUTPUTS]]:
                info=zipfile.ZipInfo(name,date_time=(2026,1,1,0,0,0));info.create_system=3
                info.external_attr=(stat.S_IFREG | (0o755 if name.endswith(('.dylib','/justfit-engine')) else 0o644))<<16
                info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data)
        print(json.dumps(dict(file=a.output.name,sha256=sha(a.output),metadata=meta),indent=2))
if __name__=='__main__':main()
