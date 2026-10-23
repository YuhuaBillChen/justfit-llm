"""Exercise product identity without requiring a Mac or model weights."""
from pathlib import Path
import os
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(ROOT / 'source'))


class BrandingTests(unittest.TestCase):
    def test_runtime_installer_and_cli_agree_on_new_output_names(self):
        from justfit_llm.cli import BUILD_OUTPUTS
        from justfit_llm.runtime_bundle import OUTPUTS
        self.assertEqual(BUILD_OUTPUTS, OUTPUTS)
        self.assertEqual(OUTPUTS[:3], ('source/build/justfit-engine',
            'source/build/justfit.metallib', 'source/build/justfit-verify16.metallib'))

    def test_source_launcher_reports_justfit(self):
        result = subprocess.run(['sh', str(ROOT / 'source/justfit'), '--version'],
            text=True, capture_output=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('JustFit', result.stdout)
        self.assertNotIn('Splash', result.stdout)

    def test_api_metrics_have_justfit_identity(self):
        from server import metrics
        text = metrics.prometheus_metrics({'ready': True, 'memory': {'pressure': 'normal'}})
        self.assertIn('justfit_', text)
        self.assertNotIn('splash_', text)

    def test_frontend_default_starts_the_renamed_engine(self):
        from server import server
        from install import paths
        args = server.parse_args(['/model', '--tokenizer', '/tokenizer', '--model', 'owner/model'])
        command = server._native_command(args)
        self.assertEqual(Path(command[0]), paths.BINARY)
        self.assertEqual(Path(command[0]).name, 'justfit-engine')
        self.assertEqual(command[1], 'serve-native')

    def test_upstream_model_format_remains_readable(self):
        # Published weights keep their upstream repository and schema identifiers.
        from install import legacy
        text = (ROOT / 'source/install/legacy.py').read_text()
        self.assertIn('splash-packed-q4', text)
        self.assertNotIn('justfit-packed-q4', text)
        from justfit_llm.cli import DEFAULT_MODEL, DEFAULT_REVISION
        self.assertEqual(DEFAULT_MODEL, 'incoai/Qwen3.8-27B-Splash')
        self.assertEqual(DEFAULT_REVISION, '9d27070b71f7142c6b6025f03ac011d70a73cb48')


if __name__ == '__main__':
    unittest.main()
