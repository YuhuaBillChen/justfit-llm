import json, subprocess, sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Packaging(unittest.TestCase):
 def test_worker_import_does_not_reach_research_directories(self):
  code="import sys,os,json;before=set(sys.path);sys.path.insert(0,sys.argv[1]);import grouped_worker;print(json.dumps({'added_paths':[p for p in sys.path if p not in before],'bridge':str(grouped_worker.B.LIB_PATH),'torch':'torch' in sys.modules}))"
  data=json.loads(subprocess.check_output([sys.executable,'-c',code,str(ROOT/'vision-tools')],text=True))
  self.assertTrue(all(Path(p).resolve() == (ROOT/'vision-tools').resolve()
                      for p in data['added_paths']),data)
  self.assertEqual(data['bridge'],str(ROOT/'vision-tools/libcoreai_bridge.dylib'))
  self.assertFalse(data['torch'])
if __name__=='__main__':unittest.main()
