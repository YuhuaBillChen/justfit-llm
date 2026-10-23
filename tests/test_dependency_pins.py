"""Ensure the installed runtime dependency graph is explicitly pinned."""
from importlib import metadata
from pathlib import Path
import unittest
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
ROOT=Path(__file__).resolve().parents[1]

class DependencyPins(unittest.TestCase):
 def test_runtime_dependency_closure_is_pinned(self):
  pins={canonicalize_name(r.name):str(r.specifier) for line in (ROOT/'requirements-runtime.txt').read_text().splitlines() if line.strip() and not line.startswith('#') for r in [Requirement(line)]}
  environment={**default_environment(),'extra':''}
  missing=set()
  for name in pins:
   for raw in metadata.requires(name) or []:
    dependency=Requirement(raw)
    if dependency.marker is not None and not dependency.marker.evaluate(environment):continue
    if canonicalize_name(dependency.name) not in pins:missing.add(dependency.name)
  self.assertEqual(missing,set(),'unpinned runtime dependencies')
  self.assertTrue(all(v.startswith('==') and ',' not in v for v in pins.values()))

if __name__=='__main__':unittest.main()
