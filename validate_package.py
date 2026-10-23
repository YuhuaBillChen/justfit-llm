"""Verify source and helper pins in an extracted package, without Apple hardware."""
from pathlib import Path
import ast,hashlib,json,sys
ROOT=Path(__file__).resolve().parent

def audit(root):
 root=Path(root);manifest=json.loads((root/'package-manifest.json').read_text());errors=[]
 for name,want in manifest.items():
  p=root/name
  if p.is_symlink() or not p.is_file():errors.append(name+': missing or symlink');continue
  if hashlib.sha256(p.read_bytes()).hexdigest()!=want:errors.append(name+': digest mismatch')
  if p.suffix=='.py':
   try:ast.parse(p.read_text(),filename=name)
   except SyntaxError as e:errors.append(name+': '+str(e))
 return dict(passed=not errors,files=len(manifest),failures=errors)
if __name__=='__main__':
 result=audit(ROOT);print(json.dumps(result,indent=2));sys.exit(not result['passed'])
