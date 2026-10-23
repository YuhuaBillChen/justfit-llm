import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('local_launcher', ROOT / 'scripts/run_local.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LocalLauncher(unittest.TestCase):
    def test_presets_keep_language_ane_and_a8_distinct(self):
        for node in ('m4', 'm5', 'm6'):
            plan = launcher.plan(node, '/model with spaces', 12345, '/state with spaces')
            self.assertEqual('--disable-ane' in plan['command'], node != 'm6')
            self.assertEqual(plan['environment']['JUSTFIT_PREFILL_A8'], '0' if node == 'm4' else '1')
            self.assertEqual(plan['environment']['JUSTFIT_VERIFY_A8'], '0' if node == 'm4' else '1')
            self.assertEqual(plan['command'][plan['command'].index('--max-memory') + 1], '22015M')
            self.assertIn('/model with spaces', plan['command'])
            self.assertIn('/state with spaces/cache', plan['command'])
            self.assertNotIn('JUSTFIT_ANE_VISION_PROFILE', plan['environment'])
            self.assertNotIn('PYTHONPATH', plan['environment'])

    def test_execution_excludes_external_python_packages(self):
        argv = ['run_local.py', '--node', 'm5', '--model', '/model',
                '--port', '12345', '--python', sys.executable]
        with mock.patch.object(sys, 'argv', argv), \
                mock.patch.object(sys, 'platform', 'darwin'), \
                mock.patch.object(Path, 'is_file', return_value=True), \
                mock.patch.object(Path, 'is_dir', return_value=True), \
                mock.patch.object(Path, 'mkdir'), \
                mock.patch.dict(launcher.os.environ, {'PYTHONPATH': '/foreign',
                                'PYTHONHOME': '/foreign-home', 'PYTHONNOUSERSITE': '0'}, clear=True), \
                mock.patch.object(launcher.os, 'chdir'), \
                mock.patch.object(launcher.os, 'execve') as execute:
            launcher.main()
        env = execute.call_args.args[2]
        self.assertNotIn('PYTHONPATH', env)
        self.assertNotIn('PYTHONHOME', env)
        self.assertEqual(env['PYTHONNOUSERSITE'], '1')

    def test_vision_profile_is_explicit(self):
        result = launcher.plan('m6', '/model', 12345, '/state', '/assets/tiered.json')
        self.assertEqual(result['environment']['JUSTFIT_ANE_VISION_PROFILE'], '/assets/tiered.json')

    def test_dry_run_needs_no_assets_and_invalid_port_fails(self):
        command = [sys.executable, str(ROOT / 'scripts/run_local.py'), '--node', 'm5',
                   '--model', '/missing/model', '--dry-run', '--port']
        result = subprocess.run(command + ['12345'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['command'][0], str(ROOT / 'source/.venv/bin/python'))
        result = subprocess.run(command + ['0'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
