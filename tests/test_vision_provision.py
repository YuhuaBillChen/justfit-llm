"""Provisioning must not turn stale or unrelated qualification into a profile."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('vision_provision', ROOT/'vision-tools/provision.py')
provision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provision)


class Provision(unittest.TestCase):
    def test_rejects_wrong_node_and_weight_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'vision').mkdir()
            (root/'vision/model.bin').write_bytes(b'test weights')
            proof = root/'proof.json'
            module = types.SimpleNamespace(check_catalog=lambda *args: None)
            for row in [dict(node='m6', passed=True), dict(node='m4', passed=False),
                        dict(node='m4', passed=True, vision_weights_sha256='wrong')]:
                proof.write_text(json.dumps(row))
                with patch.dict(sys.modules, grouped_worker=module):
                    with self.assertRaises(ValueError):
                        provision.profile(root, root, 'm4', Path(sys.executable), proof)
                self.assertFalse((root/'qualified').exists())

    def test_rejects_unmeasured_worker_and_wrong_phases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'vision').mkdir()
            weights = root/'vision/model.bin'
            weights.write_bytes(b'test weights')
            proof = root/'proof.json'
            module = types.SimpleNamespace(check_catalog=lambda *args: None)
            for tier in [dict(worker_upper_bytes=1127*2**20+1),
                         dict(worker_upper_bytes=True),
                         dict(worker_upper_bytes=1000, phase_qualification=dict(prefill=True, decode=True))]:
                proof.write_text(json.dumps(dict(node='m6', passed=True,
                    vision_weights_sha256=provision.sha(weights), tiers={'4096': tier})))
                with patch.dict(sys.modules, grouped_worker=module):
                    with self.assertRaises(ValueError):
                        provision.profile(root, root, 'm6', Path(sys.executable), proof)

    def test_portable_graph_has_no_research_import_paths(self):
        import ast
        for name in ('vision_graph.py', 'vision_weights.py', 'provision.py'):
            text = (ROOT/'vision-tools'/name).read_text()
            ast.parse(text)
            self.assertNotIn('/Users/', text)
            self.assertNotIn('run/ane-', text)

    def test_receipt_must_bind_positions_and_catalog(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'vision').mkdir()
            (root/'vision/model.bin').write_bytes(b'weights')
            tiers = {}
            for capacity in (4096, 8192):
                tier = root/str(capacity)
                (tier/'assets').mkdir(parents=True)
                (tier/'assets/main.hash').write_bytes(b'graph')
                (tier/'positions.npy').write_bytes(b'positions')
                (tier/'catalog.json').write_text('{}')
                tiers[str(capacity)] = dict(worker_upper_bytes=1000,
                    phase_qualification=dict(prefill=True, decode=True),
                    asset_files={'main.hash': provision.sha(tier/'assets/main.hash')})
            proof = root/'proof.json'
            proof.write_text(json.dumps(dict(node='m4', passed=True,
                vision_weights_sha256=provision.sha(root/'vision/model.bin'), tiers=tiers)))
            with patch.dict(sys.modules, grouped_worker=types.SimpleNamespace(check_catalog=lambda *args: None)):
                with self.assertRaisesRegex(ValueError, 'position|catalog'):
                    provision.profile(root, root, 'm4', Path(sys.executable), proof)


if __name__ == '__main__':
    unittest.main()
