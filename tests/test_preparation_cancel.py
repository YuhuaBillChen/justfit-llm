"""HTTP disconnect must reach synchronous vision preparation before admission."""
import sys,threading,time,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import test_vision_lifecycle as fixture
from server.errors import APIError
from server.preparation_cancel import submit_with_disconnect
class PreparationCancel(unittest.TestCase):
 setUp=fixture.Lifecycle.setUp
 alive=fixture.Lifecycle.alive
 released=fixture.Lifecycle.released
 zero=fixture.Lifecycle.zero
 def test_disconnect_during_sync_prepare_reaps_child(self):
  job=SimpleNamespace(image_spans=(1,),cancelled=threading.Event())
  backend=SimpleNamespace(vision_preparer=self.p,submit=lambda j:self.p.prepare(fixture.Request(),j.cancelled,time.monotonic()+5),cancel=lambda j:j.cancelled.set())
  with self.assertRaises(APIError):submit_with_disconnect(backend,job,lambda:self.pid.exists())
  self.assertTrue(job.cancelled.is_set());self.external.close.assert_called_once();self.zero()
 def test_text_submission_needs_no_disconnect_thread(self):
  calls=[];backend=SimpleNamespace(vision_preparer=self.p,submit=lambda j:calls.append(j))
  job=SimpleNamespace(image_spans=());submit_with_disconnect(backend,job,lambda:self.fail('text disconnect monitor started'));self.assertEqual(calls,[job])
if __name__=='__main__':unittest.main()
