"""Phase routing must respect language ANE ownership and unknown status."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from server import vision_routing as vr
class Relief(unittest.TestCase):
 def status(self,prefill=0,decode=0,ffn='off'):
  return dict(ready=True,ane_ffn=dict(state=ffn),scheduler=dict(prefilling=prefill,decoding=decode,queued=0))
 def route(self,s,phases=None,ahead=0,images=1):
  return vr.choose_relief_route(s,ahead,images,phases or dict(prefill=True,decode=True))
 def test_gpu_prefill_can_offload(self):self.assertEqual(self.route(self.status(prefill=1)),'ane')
 def test_language_prefill_keeps_ane(self):self.assertEqual(self.route(self.status(prefill=1,ffn='split')),'idle')
 def test_mixed_lanes_keep_language_prefill_priority(self):self.assertEqual(self.route(self.status(prefill=1,decode=2,ffn='split')),'idle')
 def test_decode_can_offload_with_language_ffn_configured(self):self.assertEqual(self.route(self.status(decode=2,ffn='split')),'ane')
 def test_no_prefill_qualification(self):self.assertEqual(self.route(self.status(prefill=1),dict(prefill=False,decode=True)),'idle')
 def test_no_decode_qualification(self):self.assertEqual(self.route(self.status(decode=1),dict(prefill=True,decode=False)),'idle')
 def test_idle_and_queued_only_use_gpu(self):
  s=self.status();self.assertEqual(self.route(s),'idle');s['scheduler']['queued']=2;self.assertEqual(self.route(s),'idle')
 def test_unknown_ffn_status_is_unavailable(self):
  s=self.status(decode=1);del s['ane_ffn'];self.assertEqual(self.route(s),'unavailable')
 def test_unknown_phase_counter_is_unavailable(self):
  for invalid in (True,-1,None,'1'):
   s=self.status(decode=1);s['scheduler']['prefilling']=invalid;self.assertEqual(self.route(s),'unavailable')
 def test_no_second_worker_or_unmeasured_multiimage(self):
  self.assertEqual(self.route(self.status(decode=1),ahead=1),'idle');self.assertEqual(self.route(self.status(decode=1),images=2),'idle')
 def test_phase_transition_is_rechecked(self):
  s=self.status(decode=1,ffn='split');self.assertEqual(self.route(s),'ane');s['scheduler']['prefilling']=1;self.assertEqual(self.route(s),'idle')
 def test_qualification_values_must_be_boolean(self):
  with self.assertRaises(ValueError):vr.validate_relief_phases(dict(prefill=1,decode=True))
if __name__=='__main__':unittest.main()
