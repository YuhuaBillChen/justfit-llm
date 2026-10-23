"""Use the existing physical/neural attribution harness for each owned worker."""
import ctypes,json,os,runpy,signal,sys,threading
from pathlib import Path
root=Path(__file__).resolve().parent
meter=root/'worker_meter.py'
sys.argv=[str(meter),str(root/'grouped_worker.py'),*sys.argv[1:]]
lib=ctypes.CDLL(str(meter.parent/'attributed_memory.dylib'))
lib.attributed_self.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
lib.attributed_pid.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64)]
stop=threading.Event()
def guard():
 while not stop.wait(.05):
  a=(ctypes.c_uint64*7)();b=(ctypes.c_uint64*4)();c=(ctypes.c_uint64*4)()
  if not lib.attributed_self(a) or not lib.attributed_pid(os.getpid(),b):continue
  bound=max(a[0],b[3])+max(a[4],a[5],b[2])
  native_pid=int(os.environ.get('JUSTFIT_RESEARCH_NATIVE_PID','0'))
  native=c[0] if native_pid and lib.attributed_pid(native_pid,c) else 0
  if bound>1127*2**20 or native+a[0]+a[4]>22015*2**20:
   path=Path(os.environ['ANE_METER_DIR'])/f'worker-guard-{os.getpid()}.json'
   path.write_text(json.dumps({'tripped':True,'worker_conservative_bound':bound,'native_physical':native,
      'worker_physical':a[0],'worker_neural_nonfootprint':a[4],'native_pid':native_pid}))
   os.kill(os.getpid(),signal.SIGTERM);return
t=threading.Thread(target=guard,daemon=True);t.start()
try:runpy.run_path(str(meter),run_name='__main__')
finally:stop.set();t.join()
