import unittest,tempfile,os
from pathlib import Path
from storage_guard import DiskPolicy,BoundedLog
class Tests(unittest.TestCase):
 def test_disk_hysteresis(self):
  p=DiskPolicy('.',20,24)
  for free,expected in [(25,True),(19,False),(21,False),(24,True),(20,True)]:self.assertEqual(p.check(free),expected)
 def test_bounded_log_preserves_tail_and_rotates(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'engine.log';l=BoundedLog(p,max_bytes=32768,backups=2)
   for i in range(80):os.write(l.fileno(),(str(i)+':'+('x'*2048)+'\n').encode())
   os.write(l.fileno(),b'FINAL_SENTINEL\n');l.close();files=list(Path(tmp).glob('engine.log*'));self.assertLessEqual(len(files),3);self.assertIn('FINAL_SENTINEL',p.read_text());self.assertLessEqual(sum(p.stat().st_size for p in files),3*(32768+16384))
if __name__=='__main__':unittest.main()
