"""Exercise the real profile loader with fake hardware, real files and pins."""
import hashlib,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from server import grouped_vision_profile as gp

class Profiles(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  self.addCleanup(self.tmp.cleanup)
 def fixture(self,capacity,count):
  r=self.root;files={}
  def put(name,data):
   p=r/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(data);files[name]=hashlib.sha256(p.read_bytes()).hexdigest()
  put('model/vision/model.bin','weights')
  for name in ['positions.bin','catalog.json','worker.py','warm_chain.py','graph.py','partition.py']:put(name,'{}')
  for i in range(count):put(f'assets/group-{i:02d}/model.aimodel/main.mlirb','graph')
  if count==27:put('assets/partition.json',json.dumps(dict(capacity=capacity,groups=[[i,i+1] for i in range(27)])))
  profile=dict(schema=2,kind='grouped-vision',capacity_raw_patches=capacity,worker_memory_bytes=1127*2**20,worker_timeout_seconds=60,vision_weights_sha256=files['model/vision/model.bin'],files=files,assets='assets',position_table='positions.bin',cache_record='catalog.json',worker='worker.py',python=sys.executable,group_count=count,route_cost_ms={})
  (r/'profile.json').write_text(json.dumps(profile));return profile
 def load(self):
  rt=Mock();worker=Mock()
  with patch.object(gp,'PreparedVision',return_value=worker),patch.object(gp,'validate_overlap_cost',return_value={}):
   result=gp.from_grouped_profile(self.root/'profile.json',rt,self.root/'model',128*2**20)
  self.assertIs(result,worker);worker.prewarm.assert_called_once();rt.reserve_external_memory.return_value.close.assert_called_once()
 def test_14400_27_loads(self):self.fixture(14400,27);self.load()
 def test_12288_27_still_loads(self):self.fixture(12288,27);self.load()
 def test_8192_9_unchanged(self):self.fixture(8192,9);self.load()
 def test_invalid_capacity_partition_rejected(self):
  self.fixture(8192,27)
  with self.assertRaisesRegex(ValueError,'partition'):self.load()
 def test_corrupt_graph_rejected(self):
  self.fixture(14400,27);(self.root/'assets/group-00/model.aimodel/main.mlirb').write_text('corrupt')
  with self.assertRaisesRegex(ValueError,'digest'):self.load()
 def test_wrong_partition_even_with_updated_pin_rejected(self):
  p=self.fixture(14400,27);name='assets/partition.json';(self.root/name).write_text(json.dumps(dict(capacity=14400,groups=[[0,27]])))
  p['files'][name]=hashlib.sha256((self.root/name).read_bytes()).hexdigest();(self.root/'profile.json').write_text(json.dumps(p))
  with self.assertRaisesRegex(ValueError,'partition'):self.load()
 def test_memory_budget_not_relaxed(self):
  p=self.fixture(14400,27);p['worker_memory_bytes']=1128*2**20;(self.root/'profile.json').write_text(json.dumps(p))
  with self.assertRaisesRegex(ValueError,'budget'):self.load()
if __name__=='__main__':unittest.main()
