"""Transient startup admission denial keeps gpu_relief service available."""
import json,unittest
from unittest.mock import Mock,patch
import test_grouped_profile as fixtures
gp=fixtures.gp
class Startup(unittest.TestCase):
 setUp=fixtures.Profiles.setUp
 fixture=fixtures.Profiles.fixture
 def test_gpu_relief_denial_defers_to_request_admission(self):
  p=self.fixture(4096,9);p.update(route_policy='gpu_relief',phase_qualification=dict(prefill=True,decode=True));(self.root/'profile.json').write_text(json.dumps(p))
  runtime=Mock();runtime.reserve_external_memory.return_value=None;worker=Mock()
  with patch.object(gp,'PreparedVision',return_value=worker):self.assertIs(gp.from_grouped_profile(self.root/'profile.json',runtime,self.root/'model',128*2**20),worker)
  worker.prewarm.assert_not_called();worker.close.assert_not_called();worker._count.assert_called_once_with('memory_unavailable')
 def test_legacy_startup_denial_still_fails(self):
  self.fixture(4096,9);runtime=Mock();runtime.reserve_external_memory.return_value=None;worker=Mock()
  with patch.object(gp,'PreparedVision',return_value=worker),patch.object(gp,'validate_overlap_cost',return_value={}):
   with self.assertRaisesRegex(RuntimeError,'startup lease denied'):gp.from_grouped_profile(self.root/'profile.json',runtime,self.root/'model',128*2**20)
  worker.close.assert_called_once()
if __name__=='__main__':unittest.main()
