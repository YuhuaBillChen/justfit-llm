"""Run the pinned node stack, rotating logs and stopping only owned groups."""
import json,os,signal,socket,subprocess,time,urllib.request
from pathlib import Path
from storage_guard import BoundedLog
D=Path(__file__).resolve().parent

def check_ports(ports):
 for port in ports:
  with socket.socket() as sock:
   sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
   sock.bind(('127.0.0.1',port))

def main():
 config=json.loads((D/'stack.json').read_text());children=[];logs=[]
 def stop(*args):raise SystemExit(0)
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 # An occupied port fails startup; the process that owns it is never stopped.
 check_ports(config['ports'])
 try:
  for entry in config['children']:
   env=dict(os.environ,**entry.get('environment',{}));log=BoundedLog(D/'logs'/(entry['name']+'.log'));logs.append(log)
   child=subprocess.Popen(entry['command'],cwd=config['repository'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children.append(child)
   if entry.get('health'):
    deadline=time.monotonic()+180
    while True:
     if any(p.poll() is not None for p in children):raise RuntimeError('owned child exited during startup')
     try:
      with urllib.request.urlopen(entry['health'],timeout=2) as response:
       if response.status==200:break
     except OSError:pass
     if time.monotonic()>deadline:raise TimeoutError(entry['name']+' readiness')
     time.sleep(.5)
  while all(p.poll() is None for p in children):
   if any(log.error for log in logs):raise RuntimeError('log writer failed')
   time.sleep(1)
  raise RuntimeError('owned serving child exited')
 finally:
  for child in reversed(children):
   if child.poll() is None:
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:pass
  for child in reversed(children):
   try:child.wait(timeout=45)
   except subprocess.TimeoutExpired:
    try:os.killpg(child.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    child.wait()
  for log in logs:log.close()
if __name__=='__main__':main()
