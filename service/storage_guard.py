"""Owned service log rotation and disk admission; never removes APC files."""
import os,threading,logging,logging.handlers,shutil
from pathlib import Path

class DiskPolicy:
 def __init__(self,path,floor=20*2**30,reopen=24*2**30):self.path=Path(path);self.floor=floor;self.reopen=reopen;self.blocked=False
 def check(self,free=None):
  free=shutil.disk_usage(self.path).free if free is None else free
  self.blocked=free<(self.reopen if self.blocked else self.floor)
  return not self.blocked

class BoundedLog:
 def __init__(self,path,max_bytes=32*2**20,backups=4):
  self.read,self.write=os.pipe();self.handler=logging.handlers.RotatingFileHandler(path,maxBytes=max_bytes,backupCount=backups,encoding='utf-8');self.handler.setFormatter(logging.Formatter('%(message)s'));self.error=None
  self.thread=threading.Thread(target=self.drain,daemon=True);self.thread.start()
 def fileno(self):return self.write
 def drain(self):
  try:
   with os.fdopen(self.read,'rb') as f:
    while True:
     data=f.read1(16384)
     if not data:break
     self.handler.emit(logging.LogRecord('engine',logging.INFO,'',0,data.decode('utf-8',errors='replace').rstrip('\n'),(),None))
  except BaseException as e:self.error=repr(e)
  finally:self.handler.close()
 def close(self):
  if self.write is not None:os.close(self.write);self.write=None
  self.thread.join(10)
  if self.thread.is_alive():raise RuntimeError('owned log pipe did not close')
  if self.error:raise RuntimeError(self.error)
