"""Keep DASH's HTTP endpoint available while releasing its GPU process on unload."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.request

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
import httpx
import uvicorn
from storage_guard import BoundedLog
from http_guard import ManagementGuard

MODEL_ID = 'incoai/Qwen3.8-27B-Splash'


class ManagedDash:
    def __init__(self, release: Path, model: Path, engine_port: int = 18141):
        self.release, self.model, self.engine_port = release, model, engine_port
        self.process = None
        self.log = None
        self.active = 0
        self.lock = threading.RLock()

    def loaded(self):
        return self.process is not None and self.process.poll() is None

    def _stop(self):
        if self.process is not None:
            # The engine owns a separate process group, including the native child.
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None

    def _start(self):
        if self.loaded():
            return
        self._stop()
        checksums = json.loads((self.release / 'sha256.json').read_text())
        for relative, expected in checksums.items():
            digest = hashlib.sha256()
            with (self.release / relative).open('rb') as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise RuntimeError(f'DASH release checksum mismatch: {relative}')
        env = {k:v for k,v in os.environ.items() if not k.startswith(('JUSTFIT_', 'DASH_', 'MLX_', 'APC_'))}
        env.update(json.loads((self.release / 'engine-env.json').read_text()))
        env.update(DASH_SOURCE=str(self.release / 'source'), DASH_PACKAGE=str(self.model),
                   DASH_APC_DIR=env.get('DASH_APC_DIR', str(Path.home() / 'Library/Caches/JustFit-Dash/dash12')))
        Path(env['DASH_APC_DIR']).mkdir(parents=True, exist_ok=True)
        logs = Path(os.environ.get('DASH_LOG_DIR', str(self.release.parent / 'logs')))
        logs.mkdir(parents=True, exist_ok=True)
        self.log = BoundedLog(logs / 'engine.log')
        command = [str(self.release / 'venv/bin/python'), '-u',
                   str(self.release / 'source/server/server.py'),
                   str(self.model / 'target'), str(self.model / 'draft'),
                   '--tokenizer', str(self.model / 'tokenizer'), '--model', MODEL_ID,
                   '--binary', str(self.release / 'source/build/justfit-engine'),
                   '--host', '127.0.0.1', '--port', str(self.engine_port),
                   '--max-context', '262144', '--max-memory', '22015M',
                   '--host-reserve', '128M', '--kv-format', 'tq4',
                   '--request-timeout', '7200', '--no-webui']
        # A JustFit release supplies its own pinned argv template (server.server, full
        # model directory, persistent-cache flags) instead of the DASH12 layout above.
        override = self.release / 'engine-command.json'
        if override.exists():
            if 'engine-command.json' not in checksums:
                raise RuntimeError('engine-command.json must be pinned in sha256.json')
            values = dict(release=str(self.release), model=str(self.model),
                          port=str(self.engine_port), cache=env['DASH_APC_DIR'])
            command = [part.format(**values) for part in json.loads(override.read_text())]
        try:
            profile = env.get('DASH_ANE_VISION_PROFILE')
            manifest = env.get('DASH_ENGINE_MANIFEST')
            if profile and not manifest:
                raise RuntimeError('ANE vision requires a pinned engine manifest')
            for flag, value in (('--ane-vision-profile', profile),
                                ('--engine-manifest', manifest)):
                if not value:
                    continue
                if not isinstance(value, str):
                    raise RuntimeError('DASH release profile paths must be strings')
                try:
                    path = (self.release / value).resolve(strict=True)
                    relative = path.relative_to(self.release.resolve()).as_posix()
                except (OSError, ValueError) as error:
                    raise RuntimeError('DASH profile must be inside the pinned release') from error
                if not path.is_file() or relative not in checksums:
                    raise RuntimeError('DASH profile must be pinned in sha256.json')
                command.extend((flag, str(path)))
            self.process = subprocess.Popen(command, cwd=self.release / 'source', env=env,
                                            stdout=self.log, stderr=subprocess.STDOUT,
                                            start_new_session=True)
            deadline = time.monotonic() + 300
            while self.loaded():
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{self.engine_port}/ready', timeout=2) as response:
                        if response.status == 200:
                            return
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError('DASH readiness deadline')
                time.sleep(0.5)
            raise RuntimeError('DASH exited during startup')
        except BaseException:
            self._stop()
            raise

    def acquire(self):
        with self.lock:
            self._start()
            self.active += 1

    def release_request(self):
        with self.lock:
            self.active -= 1

    def preload(self):
        """Start the engine without serving a request (broker home restore)."""
        with self.lock:
            self._start()
        return {'status': 'success', 'loaded': self.loaded()}

    def unload(self):
        with self.lock:
            if self.active:
                raise RuntimeError('DASH has active requests')
            self._stop()
        return {'status': 'success'}

    def shutdown(self):
        with self.lock:
            self._stop()


def create_app(engine: ManagedDash, *, api_key=None, max_request_bytes=128 * 1024 * 1024):
    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(timeout=7200) as client:
            app.state.client = client
            try:
                yield
            finally:
                await asyncio.to_thread(engine.shutdown)

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(ManagementGuard,
                       api_key=api_key if api_key is not None else os.environ.get('JUSTFIT_MANAGEMENT_API_KEY'),
                       max_request_bytes=max_request_bytes)

    @app.get('/health')
    async def health():
        # Health/model polling must not cold-load the model while media is running.
        return {'status': 'ok', 'busy': engine.active > 0,
                'loaded_models': [MODEL_ID] if engine.loaded() else []}

    @app.get('/v1/models')
    async def models():
        return {'object': 'list', 'data': [{'id': MODEL_ID, 'object': 'model', 'owned_by': 'justfit'}]}

    @app.get('/v1/models/status')
    async def model_status():
        return {'models': [{'id': MODEL_ID, 'loaded': engine.loaded()}]}

    @app.post('/v1/load')
    async def load():
        try:
            return await asyncio.to_thread(engine.preload)
        except (RuntimeError, TimeoutError) as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.post('/v1/unload')
    async def unload():
        try:
            return await asyncio.to_thread(engine.unload)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.api_route('/{path:path}', methods=['POST'])
    async def proxy(path: str, request: Request):
        body = await request.body()
        # Complete acquisition even on cancellation, then release its lease.
        acquisition = asyncio.create_task(asyncio.to_thread(engine.acquire))
        try:
            await asyncio.shield(acquisition)
        except asyncio.CancelledError:
            await acquisition
            engine.release_request()
            raise
        try:
            client = request.app.state.client
            upstream = client.build_request('POST', f'http://127.0.0.1:{engine.engine_port}/{path}',
                                            content=body, headers={'Content-Type': 'application/json'})
            sending = asyncio.create_task(client.send(upstream, stream=True))
            try:
                while not sending.done():
                    done, _ = await asyncio.wait({sending}, timeout=.05)
                    if not done and await request.is_disconnected():
                        raise HTTPException(499, 'Client disconnected during preparation')
                response = await sending
            except BaseException:
                sending.cancel()
                result = (await asyncio.gather(sending, return_exceptions=True))[0]
                if not isinstance(result, BaseException):
                    await result.aclose()
                raise
        except BaseException:
            engine.release_request()
            raise

        async def stream():
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                try:
                    await response.aclose()
                finally:
                    engine.release_request()

        return StreamingResponse(stream(), status_code=response.status_code,
                                 media_type=response.headers.get('content-type', 'application/json'))

    return app


if __name__ == '__main__':
    engine = ManagedDash(Path(os.environ['DASH_RELEASE_DIR']),
                         Path(os.environ['DASH_MODEL_DIR']),
                         int(os.environ.get('DASH_NATIVE_PORT', '18141')))
    uvicorn.run(create_app(engine), host='127.0.0.1',
                port=int(os.environ.get('DASH_ENGINE_PORT', '18140')))
