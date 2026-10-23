import tempfile,unittest,subprocess,sys,time
from pathlib import Path
from unittest.mock import patch
from fastapi import HTTPException
from service import GuardedDash
class ProtectionTests(unittest.TestCase):
 def test_low_disk_denies_load(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'ane-meter').mkdir();e=GuardedDash(root,root)
   try:
    with patch.object(e.disk,'check',return_value=False),patch('service.ManagedDash._start') as start:
     with self.assertRaises(HTTPException) as err:e._start()
     self.assertEqual(err.exception.status_code,503);start.assert_not_called()
   finally:e.shutdown()
 def test_low_disk_stops_only_owned_group(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'ane-meter').mkdir();e=GuardedDash(root,root)
   owned=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
   other=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
   try:
    with patch.object(e.disk,'check',return_value=False):
     with e.lock:e.process=owned
     owned.wait(timeout=5)
     self.assertIsNone(other.poll());self.assertTrue((root/'disk-events.jsonl').exists())
   finally:e.shutdown();other.terminate();other.wait(timeout=5)
if __name__=='__main__':unittest.main()
