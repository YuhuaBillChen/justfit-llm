"""Management boundary tests; fake engine only, no native process or listener."""
import unittest
from types import SimpleNamespace

import httpx
from managed_base import create_app


class Security(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.events = []
        self.engine = SimpleNamespace(
            active=0, loaded=lambda: False,
            preload=lambda: self.events.append('load') or {'status': 'success'},
            unload=lambda: self.events.append('unload') or {'status': 'success'},
        )

    async def request(self, path='/v1/load', headers=None, body=b'', **options):
        app = create_app(self.engine, **options)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url='http://127.0.0.1:18140') as client:
            return await client.post(path, headers=headers, content=body)

    async def test_cross_origin_control_never_reaches_engine(self):
        for path in ('/v1/load', '/v1/unload', '/v1/chat/completions'):
            response = await self.request(path, headers={
                'Origin': 'https://untrusted.example',
                'Content-Type': 'application/x-www-form-urlencoded'}, body=b'x=1')
            self.assertEqual(response.status_code, 403)
        self.assertEqual(self.events, [])

    async def test_rebinding_and_browser_requests_rejected(self):
        for headers in ({'Host': 'untrusted.example'}, {'Origin': 'null'},
                        {'Sec-Fetch-Site': 'cross-site'},
                        [('Host', 'localhost'), ('Host', 'untrusted.example')]):
            self.assertEqual((await self.request(headers=headers)).status_code, 403)
        self.assertEqual(self.events, [])

    async def test_local_broker_load_unload_and_health(self):
        for path in ('/v1/load', '/v1/unload'):
            self.assertEqual((await self.request(path)).status_code, 200)
        self.assertEqual(self.events, ['load', 'unload'])
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(self.engine)),
                                    base_url='http://localhost:18140') as client:
            self.assertEqual((await client.get('/health')).status_code, 200)
            self.assertEqual((await client.get('/v1/models')).status_code, 200)
        self.assertEqual(self.events, ['load', 'unload'])

    async def test_valid_proxy_preserves_body_and_releases_lease(self):
        self.engine.engine_port = 18141
        self.engine.acquire = lambda: self.events.append('acquire')
        self.engine.release_request = lambda: self.events.append('release')
        app = create_app(self.engine)
        payload = b'{"messages": [], "stream": true}'

        def upstream(request):
            self.assertEqual(request.content, payload)
            return httpx.Response(200, content=b'data: [DONE]\n\n',
                                  headers={'content-type': 'text/event-stream'})

        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as backend:
            app.state.client = backend
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                        base_url='http://127.0.0.1:18140') as client:
                response = await client.post('/v1/chat/completions', content=payload)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content, b'data: [DONE]\n\n')
        self.assertEqual(self.events, ['acquire', 'release'])

    async def test_configured_key_required_and_duplicate_rejected(self):
        for headers in ({}, {'Authorization': 'Bearer wrong'},
                        [('Authorization', 'Bearer secret'), ('Authorization', 'Bearer secret')]):
            response = await self.request(headers=headers, api_key='secret')
            self.assertEqual(response.status_code, 401)
        self.assertEqual(self.events, [])
        response = await self.request(headers={'Authorization': 'Bearer secret'}, api_key='secret')
        self.assertEqual(response.status_code, 200)

    async def test_oversized_declared_or_streamed_body_never_loads(self):
        async def chunks():
            yield b'1234'
            yield b'56789'
        for body in (b'123456789', chunks()):
            response = await self.request('/v1/chat/completions', body=body, max_request_bytes=8)
            self.assertEqual(response.status_code, 413)
        self.assertEqual(self.events, [])

    async def test_invalid_content_length_rejected(self):
        for value in ('-1', 'abc'):
            self.assertEqual((await self.request(headers={'Content-Length': value})).status_code, 400)
        self.assertEqual(self.events, [])
