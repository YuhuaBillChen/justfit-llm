"""Behavior of the installed-source workflow, without pretending Linux runs Metal."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from justfit_llm import cli


class PipCLI(unittest.TestCase):
    def fixture(self, root):
        (root / 'source').mkdir()
        (root / 'source/a.txt').write_text('accepted source')
        (root / 'package-manifest.json').write_text(json.dumps({
            'source/a.txt': hashlib.sha256(b'accepted source').hexdigest()}))

    def test_init_verifies_payload_and_refuses_existing_destination(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'payload'; source.mkdir(); self.fixture(source)
            dest = Path(d) / 'workspace with spaces'
            with mock.patch.object(cli, 'payload_root', return_value=source):
                cli.initialize(dest)
                self.assertEqual((dest / 'source/a.txt').read_text(), 'accepted source')
                with self.assertRaises(FileExistsError): cli.initialize(dest)
            self.assertEqual((dest / 'source/a.txt').read_text(), 'accepted source')

    def test_corrupt_or_escaping_payload_never_creates_destination(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'payload'; source.mkdir(); self.fixture(source)
            dest = Path(d) / 'workspace'
            (source / 'source/a.txt').write_text('changed')
            with mock.patch.object(cli, 'payload_root', return_value=source):
                with self.assertRaises(ValueError): cli.initialize(dest)
            self.assertFalse(dest.exists())
            (source / 'package-manifest.json').write_text(json.dumps({'../outside': 'bad'}))
            with self.assertRaises(ValueError): cli.verify(source)

    def test_serve_uses_installed_interpreter_and_workspace(self):
        with mock.patch.object(cli.subprocess, 'run') as run:
            cli.main(['serve', '--project', str(ROOT), '--node', 'm5',
                      '--model', '/models/with spaces', '--port', '12345', '--dry-run'])
            argv = run.call_args.args[0]
            self.assertEqual(argv[0], sys.executable)
            self.assertIn('--python', argv)
            self.assertEqual(argv[argv.index('--python') + 1], sys.executable)
            self.assertIn('/models/with spaces', argv)
            self.assertIn('--dry-run', argv)

    def test_build_rejects_non_mac_before_running_anything(self):
        with mock.patch.object(cli.platform, 'system', return_value='Linux'), mock.patch.object(cli.subprocess, 'run') as run:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.main(['build', '--project', str(ROOT), '--metal-bin', '/missing'])
            run.assert_not_called()


class OneCommandServe(unittest.TestCase):
    GIB = 1 << 30

    def test_preset_follows_chip_generation_and_memory(self):
        self.assertEqual(cli.select_preset('Apple M4 Pro', 24 * self.GIB), 'm4')
        self.assertEqual(cli.select_preset('Apple M5 Pro', 24 * self.GIB), 'm5')
        self.assertEqual(cli.select_preset('Apple M5 Max', 64 * self.GIB), 'm5')
        self.assertEqual(cli.select_preset('Apple M6', 32 * self.GIB), 'm6')
        for brand, memory in (('Apple M3 Max', 64), ('Apple M5', 16), ('Intel(R) Core(TM) i9', 64)):
            with self.assertRaises(ValueError):
                cli.select_preset(brand, memory * self.GIB)

    def test_metal_toolchain_found_outside_xcrun(self):
        with tempfile.TemporaryDirectory() as d:
            tools = Path(d) / 'Metal.xctoolchain/usr/bin'; tools.mkdir(parents=True)
            for name in ('metal', 'metallib'): (tools / name).write_text('')
            with mock.patch.object(cli.subprocess, 'run', side_effect=cli.subprocess.CalledProcessError(1, 'xcrun')), \
                    mock.patch.object(cli, 'METAL_COMPONENT_GLOBS', (str(Path(d) / '*.xctoolchain/usr/bin'),)):
                self.assertEqual(cli.find_metal_bin(), tools)
            with mock.patch.object(cli.subprocess, 'run', side_effect=OSError), \
                    mock.patch.object(cli, 'METAL_COMPONENT_GLOBS', ()):
                self.assertIsNone(cli.find_metal_bin())

    def test_missing_toolchain_explains_how_to_install(self):
        with mock.patch.object(cli, 'verify'), mock.patch.object(cli, 'find_metal_bin', return_value=None):
            with self.assertRaisesRegex(ValueError, 'downloadComponent MetalToolchain'):
                cli.build(ROOT, None, 2)

    def test_local_model_directory_is_used_without_download(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(cli.subprocess, 'run') as run:
            self.assertEqual(cli.resolve_model(ROOT, d, None), Path(d).resolve())
            run.assert_not_called()
            with self.assertRaises(ValueError):
                cli.resolve_model(ROOT, str(Path(d) / 'missing'), None)

    def test_hub_model_is_prepared_then_linked_at_pinned_revision(self):
        done = mock.Mock(stdout='/ws/local-state/models/link\n')
        with mock.patch.object(cli.subprocess, 'run', return_value=done) as run:
            cli.resolve_model(Path('/ws'), cli.DEFAULT_MODEL, cli.DEFAULT_REVISION)
        prepare, link = (call.args[0] for call in run.call_args_list)
        self.assertEqual(prepare[-1], 'prepare'); self.assertEqual(link[-1], 'link')
        self.assertIn('/ws/local-state/models', prepare)
        self.assertEqual(prepare[prepare.index('--revision') + 1], cli.DEFAULT_REVISION)

    def test_bare_serve_creates_workspace_builds_downloads_and_runs(self):
        with tempfile.TemporaryDirectory() as d:
            workspace = Path(d) / 'workspace'
            with mock.patch.object(cli, 'default_project', return_value=workspace), \
                    mock.patch.object(cli, 'require_mac'), \
                    mock.patch.object(cli, 'initialize', side_effect=lambda p: p.mkdir()) as init, \
                    mock.patch.object(cli, 'verify') as verify, \
                    mock.patch.object(cli, 'detect_preset', return_value='m5'), \
                    mock.patch.object(cli, 'is_built', return_value=False), \
                    mock.patch.object(cli, 'build') as build, \
                    mock.patch.object(cli, 'resolve_model', return_value=Path('/models/pack')) as model, \
                    mock.patch.object(cli.subprocess, 'run') as run, \
                    contextlib.redirect_stderr(io.StringIO()):
                cli.main(['serve', '--project', str(workspace)])
            init.assert_called_once()
            verify.assert_called_once_with(workspace.resolve())
            build.assert_called_once()
            self.assertEqual(model.call_args.args[1:], (cli.DEFAULT_MODEL, cli.DEFAULT_REVISION))
            argv = run.call_args.args[0]
            self.assertEqual(argv[argv.index('--node') + 1], 'm5')
            self.assertEqual(argv[argv.index('--port') + 1], str(cli.DEFAULT_PORT))
            self.assertEqual(argv[argv.index('--model') + 1], '/models/pack')

    def test_built_workspace_is_not_rebuilt_and_explicit_choices_win(self):
        with mock.patch.object(cli, 'require_mac'), mock.patch.object(cli, 'is_built', return_value=True), \
                mock.patch.object(cli, 'build') as build, mock.patch.object(cli, 'detect_preset') as detect, \
                mock.patch.object(cli, 'resolve_model', return_value=Path('/m')) as model, \
                mock.patch.object(cli.subprocess, 'run') as run:
            cli.main(['serve', '--project', str(ROOT), '--node', 'm4', '--model', 'someone/Other-JustFit', '--port', '9001'])
        build.assert_not_called(); detect.assert_not_called()
        self.assertIsNone(model.call_args.args[2])
        argv = run.call_args.args[0]
        self.assertEqual(argv[argv.index('--node') + 1], 'm4')
        self.assertEqual(argv[argv.index('--port') + 1], '9001')


if __name__ == '__main__':
    unittest.main()
