"""Archive/cache safety on toy files; real ANE execution is a separate Mac gate."""
from pathlib import Path
import contextlib,hashlib,io,json,os,stat,sys,tempfile,unittest,warnings,zipfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from justfit_llm import ane_bundle as a,cli

class AneBundle(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.base=Path(self.temp.name).resolve()
  self.root=self.base/'workspace';(self.root/'vision-tools').mkdir(parents=True)
  (self.root/'vision-tools/worker.py').write_text('fixture')
  (self.root/'build_release.py').write_text('helper build contract')
  self.model=self.base/'model';(self.model/'vision').mkdir(parents=True);(self.model/'vision/model.bin').write_bytes(b'weights')
  self.cache=self.base/'fresh-cache';self.payload={'positions.npy':b'positions','LICENSE':b'Apache','NOTICE':b'attribution'};groups=[]
  for i in range(9):
   raw=bytes([i])*32;key=raw.hex();groups.append(dict(asset_hash=key,cache_relative=f'26A434/org.python.python/{key}/{"A"*64}/model.aimodelx'))
   self.payload[f'assets/group-{i:02d}/model.aimodel/main.hash']=raw
   self.payload[f'compiled/group-{i:02d}/program.bin']=b'compiled-'+raw
  self.meta=dict(schema=1,kind='justfit-ane-vision',node='m4',capacity=4096,architecture='arm64',chip='Apple M4 Pro',os_build='26A434',vision_weights_sha256=a.sha(self.model/'vision/model.bin'),worker_files=a.worker_files(self.root,include_binaries=False),helper_abi=1,helper_build_contract_sha256=a.sha(self.root/'build_release.py'),groups=groups,reference_qualification_sha256='1'*64,reference_qualification=dict(passed=True,node='m4',worker_upper_bytes=100,phase_qualification=dict(prefill=True,decode=True)),files={n:hashlib.sha256(v).hexdigest() for n,v in self.payload.items()})
  p=patch.object(a,'identity',return_value=dict(chip='Apple M4 Pro',os_build='26A434'));p.start();self.addCleanup(p.stop)
  ns=patch.object(a,'cache_namespace',return_value='org.python.python',create=True);ns.start();self.addCleanup(ns.stop)
 def archive(self,extra=None,symlink=False,corrupt=False):
  path=self.base/'bundle.zip'
  with zipfile.ZipFile(path,'w') as z:
   for n,data in [('bundle.json',json.dumps(self.meta).encode()),*self.payload.items(),*(extra or [])]:
    i=zipfile.ZipInfo(n);i.create_system=3;i.external_attr=((stat.S_IFLNK if symlink else stat.S_IFREG)|0o600)<<16
    z.writestr(i,b'corrupt' if corrupt and n=='positions.npy' else data)
  return path
 def install(self,path):return a.install(self.root,path,a.sha(path),'m4',self.model,cache_root=self.cache)
 def test_install_and_idempotent_reuse(self):
  path=self.archive();self.install(path);self.install(path)
  self.assertEqual((self.cache/self.meta['groups'][0]['cache_relative']/'program.bin').read_bytes(),self.payload['compiled/group-00/program.bin'])
  dest=self.root/'local-state/ane/m4/4096';self.assertTrue((dest/'bundle.json').is_file());self.assertFalse((dest/'compiled').exists())
 def test_reuse_rejects_changed_cache_even_with_preserved_stat(self):
  archive=self.archive();self.install(archive)
  target=self.cache/self.meta['groups'][0]['cache_relative']/'program.bin';st=target.stat()
  target.write_bytes(b'X'*st.st_size);os.utime(target,ns=(st.st_atime_ns,st.st_mtime_ns))
  with self.assertRaises(ValueError):self.install(archive)
  self.assertEqual(target.read_bytes(),b'X'*st.st_size)
 def test_reuse_rejects_missing_and_extra_cache_members(self):
  archive=self.archive();self.install(archive)
  target=self.cache/self.meta['groups'][0]['cache_relative']/'program.bin';original=target.read_bytes();target.unlink()
  with self.assertRaises(ValueError):self.install(archive)
  self.assertFalse(target.exists());target.write_bytes(original)
  extra=target.parent/'unexpected';extra.write_bytes(b'keep')
  with self.assertRaises(ValueError):self.install(archive)
  self.assertEqual(extra.read_bytes(),b'keep')
 def test_reuse_rejects_cache_and_asset_symlinks(self):
  archive=self.archive();self.install(archive)
  for target in [self.cache/self.meta['groups'][0]['cache_relative']/'program.bin',self.root/'local-state/ane/m4/4096/positions.npy']:
   data=target.read_bytes();outside=self.base/'outside';outside.write_bytes(data);target.unlink();target.symlink_to(outside)
   with self.subTest(target=target),self.assertRaises(ValueError):self.install(archive)
   self.assertEqual(outside.read_bytes(),data);target.unlink();target.write_bytes(data)
 def test_reuse_rejects_special_cache_member_without_reading(self):
  archive=self.archive();self.install(archive)
  target=self.cache/self.meta['groups'][0]['cache_relative']/'extra-fifo';os.mkfifo(target)
  with self.assertRaises(ValueError):self.install(archive)
 def test_profile_checks_cache_before_starting_worker(self):
  for capacity in (4096,8192):
   self.meta['capacity']=capacity;self.install(self.archive())
  target=self.cache/self.meta['groups'][0]['cache_relative']/'program.bin';target.write_bytes(b'changed')
  with patch.object(a.Path,'home',return_value=self.base),patch.object(a,'default_cache_root',return_value=self.cache,create=True),patch.object(a.subprocess,'run') as run:
   with self.assertRaises(ValueError):a.prepare_profile(self.root,'m4',self.model)
  run.assert_not_called()
 def test_reuse_rejects_cache_directory_symlink(self):
  archive=self.archive();self.install(archive)
  target=self.cache/self.meta['groups'][0]['cache_relative'];outside=self.base/'outside-cache';target.rename(outside);target.symlink_to(outside,target_is_directory=True)
  with self.assertRaises(ValueError):self.install(archive)
  self.assertTrue((outside/'program.bin').is_file())
 def test_valid_profile_reaches_prewarm_only_after_verification(self):
  for capacity in (4096,8192):
   self.meta['capacity']=capacity;self.install(self.archive())
   (self.root/f'local-state/ane/m4/{capacity}/catalog.json').write_text('{}')
  with patch.object(a,'default_cache_root',return_value=self.cache),patch.object(a.subprocess,'run',side_effect=RuntimeError('reached prewarm')) as run:
   with self.assertRaisesRegex(RuntimeError,'reached prewarm'):a.prepare_profile(self.root,'m4',self.model)
  self.assertIn('--prewarm',run.call_args.args[0])
 def test_automatic_serve_revalidates_before_engine_launch(self):
  profile=self.root/'local-state/ane/m4/qualified/tiered.json';profile.parent.mkdir(parents=True);profile.write_text('{}')
  with patch.object(cli,'require_mac'),patch.object(cli,'verify'),patch.object(cli,'is_built',return_value=True),patch.object(cli,'resolve_model',return_value=self.model),patch.object(cli,'install_ane_assets',return_value=None),patch.object(a,'prepare_profile',side_effect=ValueError('corrupt cache')) as prepare,patch.object(cli.subprocess,'run') as run,contextlib.redirect_stderr(io.StringIO()):
   with self.assertRaises(SystemExit):cli.main(['serve','--project',str(self.root),'--node','m4','--model',str(self.model)])
  prepare.assert_called_once();run.assert_not_called()
 def test_standalone_python_cache_installs_into_current_namespace(self):
  for row in self.meta['groups']:
   row['cache_relative']=row['cache_relative'].replace('/org.python.python/','/python/')
  self.install(self.archive())
  rel=self.meta['groups'][0]['cache_relative'].replace('/python/','/org.python.python/')
  self.assertTrue((self.cache/rel/'program.bin').is_file())
 def test_python_bundle_cache_installs_into_standalone_namespace(self):
  with patch.object(a,'cache_namespace',return_value='python'):
   self.install(self.archive())
  rel=self.meta['groups'][0]['cache_relative'].replace('/org.python.python/','/python/')
  self.assertTrue((self.cache/rel/'program.bin').is_file())
 def test_existing_different_cache_is_never_overwritten(self):
  target=self.cache/self.meta['groups'][0]['cache_relative'];target.mkdir(parents=True);(target/'program.bin').write_bytes(b'user-data')
  with self.assertRaisesRegex(ValueError,'no cache was overwritten'):self.install(self.archive())
  self.assertEqual((target/'program.bin').read_bytes(),b'user-data')
 def test_wrong_hash_does_not_create_cache(self):
  path=self.archive()
  with self.assertRaisesRegex(ValueError,'SHA256'):a.install(self.root,path,'0'*64,'m4',self.model,cache_root=self.cache)
  self.assertFalse(self.cache.exists())
 def test_wrong_chip_and_os_rejected(self):
  for key,value in [('chip','Apple M5 Pro'),('os_build','other')]:
   old=self.meta[key];self.meta[key]=value
   with self.subTest(key=key),self.assertRaisesRegex(ValueError,'chip and macOS'):self.install(self.archive())
   self.meta[key]=old
 def test_mismatched_weights_and_helpers_rejected(self):
  for field in ['vision_weights_sha256','worker_files']:
   old=self.meta[field];self.meta[field]='0'*64 if field.startswith('vision') else {}
   with self.subTest(field=field),self.assertRaises(ValueError):self.install(self.archive())
   self.meta[field]=old
 def test_duplicate_traversal_and_extra_members_rejected(self):
  with warnings.catch_warnings():
   warnings.simplefilter('ignore',UserWarning)
   for extra in [[('bundle.json',b'{}')],[('../escape',b'x')],[('script.py',b'run')]]:
    with self.subTest(extra=extra),self.assertRaises(ValueError):self.install(self.archive(extra))
  self.assertFalse((self.base/'escape').exists())
 def test_symlink_archive_rejected(self):
  with self.assertRaises(ValueError):self.install(self.archive(symlink=True))
 def test_corrupt_member_never_publishes_cache(self):
  with self.assertRaisesRegex(ValueError,'member SHA256'):self.install(self.archive(corrupt=True))
  self.assertFalse((self.cache/self.meta['groups'][0]['cache_relative']).exists())
 def test_cache_namespace_cannot_escape(self):
  self.meta['groups'][0]['cache_relative']='../../escape'
  with self.assertRaisesRegex(ValueError,'namespace'):self.install(self.archive())
 def test_destination_symlink_is_rejected(self):
  (self.root/'local-state').symlink_to(self.base,target_is_directory=True)
  with self.assertRaisesRegex(ValueError,'symlink'):self.install(self.archive())
 def test_exclusive_directory_publication_preserves_destination(self):
  src=self.base/'src';dst=self.base/'dst';src.mkdir();dst.mkdir();(dst/'owned').write_text('keep')
  with self.assertRaises(FileExistsError):a.rename_exclusive(src,dst)
  self.assertEqual((dst/'owned').read_text(),'keep');self.assertTrue(src.exists())
 def test_gpu_override_does_not_select_an_installed_profile(self):
  profile=self.root/'local-state/ane/m4/qualified/tiered.json';profile.parent.mkdir(parents=True);profile.write_text('{}')
  with patch.object(cli,'verify'),patch.object(cli.subprocess,'run') as run:
   cli.main(['serve','--project',str(self.root),'--node','m4','--model','/model','--no-ane-vision','--dry-run'])
  self.assertNotIn('--vision-profile',run.call_args.args[0])
 def test_signature_envelope_does_not_change_graph_source_contract(self):
  helper=self.root/'vision-tools/bridge.dylib';helper.write_bytes(b'old signature')
  self.meta['reference_native_helpers']={'bridge.dylib':a.sha(helper)}
  helper.write_bytes(b'new signature')
  a.validate_metadata(self.meta,self.root,'m4',self.model)
 def test_helper_source_and_abi_changes_require_new_qualification(self):
  for field,value in [('helper_abi',99),('helper_build_contract_sha256','0'*64)]:
   old=self.meta[field];self.meta[field]=value
   with self.subTest(field=field),self.assertRaisesRegex(ValueError,'source and ABI'):
    a.validate_metadata(self.meta,self.root,'m4',self.model)
   self.meta[field]=old
 def test_http_index_is_rejected_before_workspace_creation(self):
  with patch.object(cli,'require_mac'),patch.object(cli,'initialize') as init,contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
   cli.main(['serve','--project',str(self.base/'new'),'--node','m4','--ane-manifest','http://example.invalid/index','--ane-manifest-sha256','1'*64])
  init.assert_not_called()
 def test_missing_ane_hash_fails_before_workspace_creation(self):
  with patch.object(cli,'initialize') as init,contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
   cli.main(['serve','--project',str(self.base/'new'),'--node','m4','--ane-bundle',str(self.base/'x'),'--dry-run'])
  init.assert_not_called()

if __name__=='__main__':unittest.main()
