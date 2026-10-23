"""First-run validation and reuse must happen before expensive or executable work."""
import contextlib,hashlib,io,json,sys,tempfile,unittest
from pathlib import Path
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from justfit_llm import cli

class FirstRunSafety(unittest.TestCase):
 def test_complete_runtime_checks_coreai_os_floor(self):
  with mock.patch.multiple(cli.platform,system=lambda:'Darwin',machine=lambda:'arm64',mac_ver=lambda:('26.4',(),'')):
   with self.assertRaisesRegex(ValueError,'27.0'):cli.require_mac()
  with mock.patch.multiple(cli.platform,system=lambda:'Darwin',machine=lambda:'arm64',mac_ver=lambda:('27.0.1',(),'')):
   cli.require_mac()
 def test_unknown_future_chip_requires_explicit_selection(self):
  with self.assertRaises(ValueError):cli.select_preset('Apple M99 Pro',64*(1<<30))
 def test_invalid_port_fails_before_creating_or_building(self):
  with tempfile.TemporaryDirectory() as d, mock.patch.object(cli,'require_mac'), mock.patch.object(cli,'initialize') as init, mock.patch.object(cli,'build') as build:
   with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
    cli.main(['serve','--project',str(Path(d)/'new'),'--node','m4','--port','0'])
   init.assert_not_called();build.assert_not_called()
 def test_existing_corrupt_workspace_never_runs(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'source').mkdir();(p/'source/a.txt').write_text('corrupt')
   (p/'package-manifest.json').write_text(json.dumps({'source/a.txt':hashlib.sha256(b'expected').hexdigest()}))
   with mock.patch.object(cli,'require_mac'),mock.patch.object(cli,'is_built',return_value=True),mock.patch.object(cli,'resolve_model') as model,mock.patch.object(cli.subprocess,'run') as run, contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
    cli.main(['serve','--project',str(p),'--node','m5'])
   model.assert_not_called();run.assert_not_called()
 def test_cached_build_requires_matching_receipt_and_intact_outputs(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'package-manifest.json').write_text('{}')
   for name in cli.BUILD_OUTPUTS:
    f=p/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b'compiled')
   self.assertFalse(cli.is_built(p))
   receipt={'source_manifest_sha256':hashlib.sha256(b'{}').hexdigest(),'outputs':{n:hashlib.sha256(b'compiled').hexdigest() for n in cli.BUILD_OUTPUTS}}
   (p/'build-receipt.json').write_text(json.dumps(receipt))
   self.assertTrue(cli.is_built(p))
   (p/cli.BUILD_OUTPUTS[0]).write_bytes(b'corrupt')
   self.assertFalse(cli.is_built(p))

if __name__=='__main__':unittest.main()
