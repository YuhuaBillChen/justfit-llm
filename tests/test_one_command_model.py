"""Exercise the real installer revision contract used by the one-command CLI."""
import sys,tempfile,unittest
from pathlib import Path
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from install import legacy,models

REV='9d27070b71f7142c6b6025f03ac011d70a73cb48'
class PinnedRuntimePack(unittest.TestCase):
 def test_fixed_revision_pack_reaches_verified_installation(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);snapshot=root/'snapshot';snapshot.mkdir()
   selection=models.Selection.of(root/'models','incoai/Qwen3.8-27B-Splash',revision=REV)
   with mock.patch.object(legacy,'verify',side_effect=[models.ModelError('missing'),None]), \
        mock.patch.object(legacy,'resolve_snapshot',return_value=snapshot) as download, \
        mock.patch.object(legacy.hub,'pin',return_value='pin'), \
        mock.patch.object(legacy.hub,'retire_other_pins'), \
        mock.patch.object(models,'link_selection'):
    legacy.prepare(selection)
   download.assert_called_once_with(selection.repo_id,revision=REV)


class ExactRevisionCache(unittest.TestCase):
 def test_cached_snapshot_looks_up_requested_revision(self):
  import types
  fake=types.ModuleType('huggingface_hub')
  fake.try_to_load_from_cache=mock.Mock(return_value=None)
  with mock.patch.dict(sys.modules,{'huggingface_hub':fake}):
   self.assertIsNone(legacy._cached_snapshot('incoai/Qwen3.8-27B-Splash',revision=REV))
  fake.try_to_load_from_cache.assert_called_once_with('incoai/Qwen3.8-27B-Splash','manifest.json',revision=REV)
 def test_cached_snapshot_of_other_commit_is_rejected(self):
  import types
  fake=types.ModuleType('huggingface_hub')
  fake.try_to_load_from_cache=mock.Mock(return_value='/cache/snapshots/other/manifest.json')
  with mock.patch.dict(sys.modules,{'huggingface_hub':fake}),mock.patch.object(legacy.hub,'snapshot_commit',return_value='0'*40):
   self.assertIsNone(legacy._cached_snapshot('incoai/Qwen3.8-27B-Splash',revision=REV))

class ExactRevisionDownload(unittest.TestCase):
 def test_hub_cannot_resolve_pin_to_another_commit(self):
  import types
  fake=types.ModuleType('huggingface_hub')
  fake.HfApi=mock.Mock()
  fake.HfApi.return_value.model_info.return_value=types.SimpleNamespace(sha='0'*40,siblings=[])
  fake.hf_hub_download=mock.Mock();fake.snapshot_download=mock.Mock()
  with mock.patch.dict(sys.modules,{'huggingface_hub':fake}),self.assertRaisesRegex(models.ModelError,'requested commit'):
   legacy._download_snapshot('incoai/Qwen3.8-27B-Splash',None,requested_revision=REV)
  fake.hf_hub_download.assert_not_called();fake.snapshot_download.assert_not_called()

if __name__=='__main__':unittest.main()
