"""Bundle the reviewed source manifest as wheel data; never compile at pip install."""
import hashlib
import json
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

ROOT = Path(__file__).resolve().parent


class BuildWithSources(build_py):
    def run(self):
        super().run()
        manifest = json.loads((ROOT / 'package-manifest.json').read_text())
        for name, digest in manifest.items():
            relative = Path(name)
            source = ROOT / relative
            if (relative.is_absolute() or '..' in relative.parts or source.is_symlink()
                    or not source.resolve().is_relative_to(ROOT) or not source.is_file()):
                raise ValueError('unsafe or missing package source: ' + name)
            if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise ValueError('stale source manifest; regenerate the project archive: ' + name)
        for name in [*manifest, 'package-manifest.json']:
            target = Path(self.build_lib) / 'justfit_llm/project' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            self.copy_file(str(ROOT / name), str(target))


class PublicSdist(sdist):
    def make_distribution(self):
        # Do not expose the build account in POSIX tar headers.
        self.owner = 'root'
        self.group = 'root'
        super().make_distribution()


setup(cmdclass={'build_py': BuildWithSources, 'sdist': PublicSdist})
