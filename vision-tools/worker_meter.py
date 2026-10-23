import ctypes,json,os,runpy,sys,threading,time
from pathlib import Path
root=Path(__file__).resolve().parent
worker,*args=sys.argv[1:]
sys.argv=[worker,*args]
sys.path.insert(0,str(Path(worker).parent))
lib=ctypes.CDLL(str(root/'attributed_memory.dylib'))
lib.attributed_self.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
lib.attributed_pid.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64)]
keys=['phys_footprint','graphics_footprint','graphics_nofootprint','neural_footprint','neural_nofootprint','neural_nofootprint_peak','neural_nofootprint_total']
log=open(Path(os.environ['ANE_METER_DIR'])/f'worker-memory-{os.getpid()}.jsonl','x',buffering=1)
def sample():
 a=(ctypes.c_uint64*7)();b=(ctypes.c_uint64*4)()
 ok=bool(lib.attributed_self(a));ok2=bool(lib.attributed_pid(os.getpid(),b))
 log.write(json.dumps(dict(time=time.monotonic(),pid=os.getpid(),ok=ok,pid_ok=ok2,ledger=dict(zip(keys,a)),rusage=list(b)))+'\n')
stop=threading.Event()
def monitor():
 while not stop.wait(.05):sample()
sample();thread=threading.Thread(target=monitor,daemon=True);thread.start()
try:runpy.run_path(worker,run_name='__main__')
finally:stop.set();thread.join();sample();log.close()
