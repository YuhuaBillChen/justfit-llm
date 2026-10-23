"""Managed JustFit service with owned-process disk and log protection."""
from pathlib import Path
import os,sys,json,threading,time,hashlib
import uvicorn
from fastapi import HTTPException
from storage_guard import DiskPolicy
from managed_base import ManagedDash,create_app
D=Path(__file__).resolve().parent
class GuardedDash(ManagedDash):
 def __init__(self,*args,**kw):
  super().__init__(*args,**kw);self.disk=DiskPolicy(self.release);self.stop_guard=threading.Event();self.guard_error=None
  self.watcher=threading.Thread(target=self.watch,daemon=True);self.watcher.start()
 def _start(self):
  if self.guard_error:raise HTTPException(503,self.guard_error)
  if sum(p.stat().st_size for p in (self.release/'ane-meter').glob('*') if p.is_file())>128*2**20:raise HTTPException(503,'ANE diagnostic log budget reached; archive logs before loading')
  if not self.disk.check():raise HTTPException(503,'Disk reserve below threshold; engine admission paused')
  super()._start()
 def watch(self):
  while not self.stop_guard.wait(1):
   try:
    with self.lock:
     meter_full=sum(p.stat().st_size for p in (self.release/'ane-meter').glob('*') if p.is_file())>128*2**20
     if (not self.disk.check() or meter_full) and self.loaded():
      with (self.release/'disk-events.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),action='stop_owned_engine',reason='ANE log budget exceeded' if meter_full else 'free space below20GiB',pid=self.process.pid))+'\n')
      self._stop()
   except BaseException as e:
    self.guard_error='disk protection failed: '+repr(e)
    with self.lock:self._stop()
    return
 def shutdown(self):
  self.stop_guard.set();super().shutdown();self.watcher.join(5)

def main():
 config=json.loads((D/'service.json').read_text())
 for n,h in json.loads((D/'sha256.json').read_text()).items():
  if hashlib.sha256((D/n).read_bytes()).hexdigest()!=h:raise RuntimeError('release pin mismatch: '+n)
 os.environ['DASH_RELEASE_DIR']=str(D);os.environ['DASH_LOG_DIR']=str(D/'logs');(D/'logs').mkdir(exist_ok=True)
 engine=GuardedDash(D,Path(config['model']),int(os.environ.get('JUSTFIT_ACCEPT_NATIVE_PORT',config['native_port'])))
 app=create_app(engine)
 uvicorn.run(app,host='127.0.0.1',port=int(os.environ.get('JUSTFIT_ACCEPT_HTTP_PORT',config['http_port'])),access_log=False,log_level='warning')
if __name__=='__main__':main()
