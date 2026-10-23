import asyncio,unittest
from types import SimpleNamespace
from fastapi import HTTPException
from starlette.requests import Request
from managed_base import create_app
class Disconnect(unittest.IsolatedAsyncioTestCase):
 async def test_disconnect_before_upstream_headers_cancels_and_releases(self):
  cancelled=asyncio.Event();disconnect=asyncio.Event();counts=dict(acquire=0,release=0)
  engine=SimpleNamespace(engine_port=1,acquire=lambda:counts.__setitem__('acquire',counts['acquire']+1),release_request=lambda:counts.__setitem__('release',counts['release']+1))
  class Client:
   def build_request(self,*a,**kw):return None
   async def send(self,*a,**kw):
    disconnect.set()
    try:await asyncio.Event().wait()
    except asyncio.CancelledError:cancelled.set();raise
  app=create_app(engine);app.state.client=Client();sent=False
  async def receive():
   nonlocal sent
   if not sent:sent=True;return dict(type='http.request',body=b'{}',more_body=False)
   await disconnect.wait();return dict(type='http.disconnect')
  request=Request(dict(type='http',method='POST',path='/v1/chat/completions',headers=[],app=app),receive)
  proxy=next(r.endpoint for r in app.routes if getattr(r,'path',None)=='/{path:path}')
  with self.assertRaises(HTTPException) as caught:await asyncio.wait_for(proxy('v1/chat/completions',request),timeout=.5)
  self.assertEqual(caught.exception.status_code,499);self.assertTrue(cancelled.is_set());self.assertEqual(counts,dict(acquire=1,release=1))
 async def test_normal_response_releases_once_after_stream(self):
  counts=dict(acquire=0,release=0,close=0)
  engine=SimpleNamespace(engine_port=1,acquire=lambda:counts.__setitem__('acquire',counts['acquire']+1),release_request=lambda:counts.__setitem__('release',counts['release']+1))
  class Response:
   status_code=200;headers={'content-type':'application/json'}
   async def aiter_bytes(self):yield b'{}'
   async def aclose(self):counts['close']+=1
  class Client:
   def build_request(self,*a,**kw):return None
   async def send(self,*a,**kw):return Response()
  app=create_app(engine);app.state.client=Client()
  async def receive():return dict(type='http.request',body=b'{}',more_body=False)
  request=Request(dict(type='http',method='POST',path='/v1/chat/completions',headers=[],app=app),receive)
  proxy=next(r.endpoint for r in app.routes if getattr(r,'path',None)=='/{path:path}')
  response=await proxy('v1/chat/completions',request)
  self.assertEqual(counts['release'],0)
  self.assertEqual([part async for part in response.body_iterator],[b'{}'])
  self.assertEqual(counts,dict(acquire=1,release=1,close=1))
if __name__=='__main__':unittest.main()
