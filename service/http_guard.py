"""Machine-only loopback boundary for the optional management service."""
import asyncio
import hmac
from urllib.parse import urlsplit

from starlette.responses import JSONResponse


class ManagementGuard:
    def __init__(self, app, api_key=None, max_request_bytes=128 * 1024 * 1024):
        if api_key is not None and (not api_key or any(ord(c) <= 32 or ord(c) >= 127 for c in api_key)):
            raise ValueError('management API key must be nonempty visible ASCII')
        if max_request_bytes < 1:
            raise ValueError('max_request_bytes must be positive')
        self.app, self.api_key, self.limit = app, api_key, max_request_bytes

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)

        async def reject(code, message):
            await JSONResponse({'error': message}, status_code=code)(scope, receive, send)

        headers = {}
        for name, value in scope['headers']:
            headers.setdefault(name.lower(), []).append(value)
        hosts = headers.get(b'host', [])
        try:
            authority = hosts[0].decode('ascii') if len(hosts) == 1 else ''
            parsed = urlsplit('//' + authority)
            port = parsed.port
            valid = (parsed.hostname in ('localhost', '127.0.0.1', '::1')
                     and not parsed.username and not parsed.password
                     and not parsed.path and not parsed.query and not parsed.fragment
                     and not any(c.isspace() for c in authority)
                     and (port is None or 1 <= port <= 65535))
        except (ValueError, UnicodeError):
            valid = False
        # This service has no browser UI. Reject even same-origin browser writes;
        # ordinary local broker clients send neither Origin nor Fetch Metadata.
        if not valid or b'origin' in headers or b'sec-fetch-site' in headers:
            return await reject(403, 'management endpoint accepts local machine clients only')
        if self.api_key is not None:
            supplied = headers.get(b'authorization', [])
            expected = ('Bearer ' + self.api_key).encode('ascii')
            if len(supplied) != 1 or not hmac.compare_digest(supplied[0], expected):
                return await reject(401, 'invalid or missing management API key')

        lengths = headers.get(b'content-length', [])
        if lengths:
            if len(lengths) != 1 or not lengths[0].isdigit() or len(lengths[0]) > 20:
                return await reject(400, 'invalid Content-Length')
            if b'transfer-encoding' in headers:
                return await reject(400, 'ambiguous request framing')
            if int(lengths[0]) > self.limit:
                return await reject(413, 'request body too large')
        body = bytearray()
        try:
            async with asyncio.timeout(30):
                while True:
                    message = await receive()
                    if message['type'] == 'http.disconnect':
                        return
                    part = message.get('body', b'')
                    if len(body) + len(part) > self.limit:
                        return await reject(413, 'request body too large')
                    body.extend(part)
                    if not message.get('more_body', False):
                        break
        except TimeoutError:
            return await reject(408, 'request upload timed out')
        if lengths and len(body) != int(lengths[0]):
            return await reject(400, 'Content-Length mismatch')

        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
            return await receive()

        await self.app(scope, bounded_receive, send)
