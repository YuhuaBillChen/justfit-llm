import hashlib,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from validate_package import audit
class Integrity(unittest.TestCase):
 def test_missing_corrupted_syntax_and_symlink_fail(self):
  with tempfile.TemporaryDirectory() as temp:
   r=Path(temp);p=r/'a.py';p.write_text('x=1\n')
   def pin():(r/'package-manifest.json').write_text(json.dumps({'a.py':hashlib.sha256(p.read_bytes()).hexdigest()}))
   pin();self.assertTrue(audit(r)['passed'])
   p.write_text('x=2\n');self.assertFalse(audit(r)['passed'])
   p.unlink();self.assertFalse(audit(r)['passed'])
   p.write_text('if');pin();self.assertFalse(audit(r)['passed'])
   p.unlink();q=r/'b.py';q.write_text('x=1\n');p.symlink_to(q);pin();self.assertFalse(audit(r)['passed'])
if __name__=='__main__':unittest.main()
