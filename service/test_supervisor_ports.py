import importlib.util
from pathlib import Path
import socket
import unittest

spec = importlib.util.spec_from_file_location('supervisor', Path(__file__).with_name('supervisor.py'))
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)

class RestartPorts(unittest.TestCase):
 def test_active_listener_is_never_taken(self):
  with socket.socket() as server:
   server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
   server.bind(('127.0.0.1',0));server.listen()
   with self.assertRaises(OSError):supervisor.check_ports([server.getsockname()[1]])
 def test_recent_owned_connection_does_not_block_restart(self):
  server=socket.socket();server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
  server.bind(('127.0.0.1',0));port=server.getsockname()[1];server.listen()
  client=socket.create_connection(('127.0.0.1',port));accepted,_=server.accept()
  accepted.close();client.recv(1);client.close();server.close()
  supervisor.check_ports([port])
if __name__=='__main__':unittest.main()
