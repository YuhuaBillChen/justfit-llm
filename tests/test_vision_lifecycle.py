"""Real child lifecycle: cancellation/failure reaps before releasing its lease."""
import json,os,sys,tempfile,threading,time,unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from server.prepared_vision import PreparedVision,PayloadBudget
from server.errors import APIError
@dataclass(frozen=True)
class Span:
 grid_height:int=2;grid_width:int=2;tokens:int=1;digest_lo:int=0;digest_hi:int=0
@dataclass(frozen=True)
class Request:
 image_spans:tuple=(Span(),);image_pixels:bytes=bytes(4*768);image_embedding_width:int=0;image_owner:object=None
class Lifecycle(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.pid=self.root/'pid'
  self.worker=self.root/'worker.py';self.worker.write_text('import os,sys,time\nfrom pathlib import Path\nPath(sys.argv[1]).write_text(str(os.getpid()))\ntime.sleep(60)\n')
  self.external=Mock();self.external.close.side_effect=self.released
  self.p=PreparedVision([sys.executable,str(self.worker),str(self.pid)],bytes(32),lambda:True,width=2,capacity_raw_patches=4096,reserve=lambda:self.external,route=lambda a,i:'ane')
  self.p.shared_payload_budget=PayloadBudget(1024);self.addCleanup(self.p.close)
 def alive(self):
  if not self.pid.exists():return False
  try:os.kill(int(self.pid.read_text()),0);return True
  except ProcessLookupError:return False
 def released(self):self.assertFalse(self.alive(),'lease released before child reaped')
 def zero(self):
  self.assertEqual(self.p.snapshot()['payload_reserved_bytes'],0);self.assertEqual(self.p.snapshot()['queued_images'],0);self.assertEqual(self.p.shared_payload_budget.snapshot()['reserved_bytes'],0)
 def test_denial_falls_back_without_worker(self):
  self.p.reserve=lambda:None;r=Request();self.assertIs(self.p.prepare(r,threading.Event(),time.monotonic()+5),r);self.assertFalse(self.pid.exists());self.zero()
 def test_phase_changes_while_waiting_lease(self):
  def reserve():self.p.route=lambda a,i:'idle';return self.external
  self.p.reserve=reserve;r=Request();self.assertIs(self.p.prepare(r,threading.Event(),time.monotonic()+5),r);self.external.close.assert_called_once();self.assertFalse(self.pid.exists());self.zero()
 def test_cancellation_reaps_child_then_releases(self):
  cancel=threading.Event();errors=[]
  def run():
   try:self.p.prepare(Request(),cancel,time.monotonic()+10)
   except APIError as e:errors.append(e)
  t=threading.Thread(target=run);t.start()
  until=time.monotonic()+5
  while not self.pid.exists() and time.monotonic()<until:time.sleep(.01)
  self.assertTrue(self.pid.exists());cancel.set();t.join(5);self.assertFalse(t.is_alive());self.assertEqual(len(errors),1);self.external.close.assert_called_once();self.zero()
 def test_timeout_falls_back_and_recovers(self):
  self.p.worker_timeout=.1;r=Request();self.assertIs(self.p.prepare(r,threading.Event(),time.monotonic()+5),r);self.external.close.assert_called_once();self.zero()
  self.worker.write_text('import sys\nfrom pathlib import Path\nPath(sys.argv[3]).write_bytes(bytes(4))\n')
  self.p.worker_timeout=5
  prepared=self.p.prepare(r,threading.Event(),time.monotonic()+5);self.assertEqual(prepared.image_embedding_width,2);prepared.image_owner[1].close();self.zero()
if __name__=='__main__':unittest.main()
