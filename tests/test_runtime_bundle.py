import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from justfit_llm import runtime_bundle as r

class RuntimeBundleTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)/'workspace';self.root.mkdir()
  (self.root/'package-manifest.json').write_text('{}')
  self.meta=dict(schema=1,version='test',architecture='arm64',minimum_macos=[27,0],source_manifest_sha256=r.sha(self.root/'package-manifest.json'),outputs={n:hashlib.sha256(n.encode()).hexdigest() for n in r.OUTPUTS})
  self.host=patch.multiple(r.platform,system=lambda:'Darwin',machine=lambda:'arm64',mac_ver=lambda:('27.0.1',(),''));self.host.start();self.addCleanup(self.host.stop)
 def archive(self,extra=None,symlink=False):
  path=Path(self.temp.name)/'runtime.zip'
  with zipfile.ZipFile(path,'w') as z:
   for n,data in [('runtime.json',json.dumps(self.meta).encode()),*[(n,n.encode()) for n in r.OUTPUTS],*(extra or [])]:
    i=zipfile.ZipInfo(n);i.create_system=3;i.external_attr=((stat.S_IFLNK if symlink else stat.S_IFREG)|0o644)<<16;z.writestr(i,data)
  return path
 def test_valid_relocatable_install_and_receipt(self):
  a=self.archive();r.install(self.root,a,r.sha(a),'test')
  self.assertEqual(json.loads((self.root/'build-receipt.json').read_text())['outputs'],self.meta['outputs'])
  self.assertTrue((self.root/r.OUTPUTS[0]).stat().st_mode & stat.S_IXUSR)
 def test_wrong_archive_hash_never_installs(self):
  a=self.archive()
  with self.assertRaisesRegex(ValueError,'archive SHA256'):r.install(self.root,a,'0'*64,'test')
  self.assertFalse((self.root/'source').exists())
 def test_reject_traversal_and_unexpected_members(self):
  a=self.archive([('../escape',b'x')])
  with self.assertRaises(ValueError):r.install(self.root,a,r.sha(a),'test')
  self.assertFalse((self.root.parent/'escape').exists())
 def test_reject_duplicate_member(self):
  a=self.archive([('runtime.json',b'{}')])
  with self.assertRaises(ValueError):r.install(self.root,a,r.sha(a),'test')
 def test_reject_symlink(self):
  a=self.archive(symlink=True)
  with self.assertRaisesRegex(ValueError,'regular'):r.install(self.root,a,r.sha(a),'test')
 def test_wrong_source_binding(self):
  self.meta['source_manifest_sha256']='0'*64;a=self.archive()
  with self.assertRaisesRegex(ValueError,'source package'):r.install(self.root,a,r.sha(a),'test')
 def test_wrong_output_hash_no_partial_install(self):
  self.meta['outputs'][r.OUTPUTS[-1]]='0'*64;a=self.archive()
  with self.assertRaisesRegex(ValueError,'output SHA256'):r.install(self.root,a,r.sha(a),'test')
  self.assertFalse((self.root/'source').exists())
 def test_wrong_version(self):
  a=self.archive()
  with self.assertRaisesRegex(ValueError,'version'):r.install(self.root,a,r.sha(a),'other')
 def test_old_os_rejected(self):
  a=self.archive()
  with patch.object(r.platform,'mac_ver',return_value=('26.4',(),'')):
   with self.assertRaisesRegex(ValueError,'newer'):r.install(self.root,a,r.sha(a),'test')
 def test_symlink_parent_rejected(self):
  (self.root/'source').symlink_to(self.root.parent,target_is_directory=True);a=self.archive()
  with self.assertRaisesRegex(ValueError,'destination'):r.install(self.root,a,r.sha(a),'test')
 def test_https_download_only(self):
  with self.assertRaisesRegex(ValueError,'HTTPS'):r.download('http://example.invalid/a',self.root/'a',100)
 def test_manifest_hash_checked_before_archive_download(self):
  def fake(url,dest,limit):dest.write_text('{}')
  with patch.object(r,'download',side_effect=fake) as download:
   with self.assertRaisesRegex(ValueError,'manifest SHA256'):r.fetch_and_install(self.root,'https://example.invalid/manifest','0'*64,'test')
   self.assertEqual(download.call_count,1)

class RedirectPolicy(unittest.TestCase):
 def test_https_downgrade_is_rejected_before_next_request(self):
  import urllib.request
  handler=r.HTTPSOnlyRedirectHandler()
  req=urllib.request.Request('https://example.invalid/asset')
  for target in ['http://example.invalid/asset','ftp://example.invalid/asset']:
   with self.subTest(target=target),self.assertRaisesRegex(ValueError,'HTTPS'):
    handler.redirect_request(req,None,302,'redirect',{},target)
 def test_https_redirect_keeps_working(self):
  import urllib.request
  handler=r.HTTPSOnlyRedirectHandler()
  result=handler.redirect_request(urllib.request.Request('https://example.invalid/start'),None,302,'redirect',{},'https://cdn.example.invalid/asset')
  self.assertEqual(result.full_url,'https://cdn.example.invalid/asset')
 def test_downloader_uses_the_strict_redirect_handler(self):
  from io import BytesIO
  from unittest.mock import MagicMock
  with tempfile.TemporaryDirectory() as temp:
   response=MagicMock();response.__enter__.return_value=response
   response.geturl.return_value='https://example.invalid/asset'
   response.read.side_effect=[b'asset',b'']
   opener=MagicMock();opener.open.return_value=response
   with patch.object(r.urllib.request,'build_opener',return_value=opener) as build:
    target=Path(temp)/'download'
    r.download('https://example.invalid/asset',target,100)
    self.assertEqual(target.read_bytes(),b'asset')
   self.assertIsInstance(build.call_args.args[0],r.HTTPSOnlyRedirectHandler)

if __name__=='__main__':unittest.main()
