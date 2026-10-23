# Modified for JustFit: product identity and integration.
"""Optional, bounded vision preprocessing before native lane admission.

A worker runs in its own process. Optional bounded warm reuse holds its
governor lease until the worker has exited, including idle residency. Other
requests may continue using the native GPU while it runs. This module never
imports Core AI, modifies the original request, or retries admitted generation.
Profile-based workers acquire a temporary native governor reservation before
loading; the ordinary OS reserve is separate from this external worker budget.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import subprocess
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path

if __package__:
    from .errors import APIError
else:
    from errors import APIError


_LOW_EXPONENT = bytes(1 if b & 0x80 else 0 for b in range(256))
_HIGH_EXPONENT = bytes(1 if b & 0x7f == 0x7f else 0 for b in range(256))


def _finite_bf16(payload):
    """Exact exponent check in bounded C-level byte operations; no dependency."""
    if len(payload) % 2:
        return False
    for offset in range(0, len(payload), 131072):
        block = payload[offset:offset + 131072]
        low = int.from_bytes(block[0::2].translate(_LOW_EXPONENT), "little")
        high = int.from_bytes(block[1::2].translate(_HIGH_EXPONENT), "little")
        if low & high:
            return False
    return True


class PayloadBudget:
    """One bound across prepared embeddings retained by different tiers."""

    def __init__(self, limit):
        self.limit = limit
        self.used = 0
        self.lock = threading.Lock()

    def reserve(self, size):
        with self.lock:
            if size > self.limit - self.used:
                return False
            self.used += size
            return True

    def release(self, size):
        with self.lock:
            self.used -= size
            if self.used < 0:
                raise RuntimeError("unbalanced vision payload ownership")

    def snapshot(self):
        with self.lock:
            return {"limit_bytes": self.limit, "reserved_bytes": self.used}


class _EmbeddingLease:
    """Keep the charge alive until the native transport releases the owner."""

    def __init__(self, owner, size):
        self.owner = owner
        self.size = size

    def close(self):
        if self.size:
            with self.owner.metrics_lock:
                self.owner.reserved_bytes -= self.size
                if self.owner.shared_payload_budget is not None:
                    self.owner.shared_payload_budget.release(self.size)
                self.size = 0

    def __del__(self):
        self.close()


class PreparedVision:
    def __init__(
        self,
        command,
        profile: bytes,
        admit,
        *,
        max_pending=4,
        worker_timeout=10.0,
        width=5120,
        max_images=4,
        payload_budget_bytes=64 * 2**20,
        reserve=None,
        reserve_wait_timeout=0.0,
        retry_reserve=None,
        grid=32,
        warm_idle_seconds=0.0,
        keep_warm=None,
        route=None,
        capacity_raw_patches=None,
    ):
        if grid not in (32, 64):
            raise ValueError("unsupported compiled grid")
        if capacity_raw_patches is not None and (
            type(capacity_raw_patches) is not int or capacity_raw_patches not in (4096, 8192, 12288, 14400, 16384)
        ):
            raise ValueError("unsupported variable-grid capacity")
        self.capacity_raw_patches = capacity_raw_patches
        self._active_grids = []
        self.grid = grid
        self.rows = grid * grid // 4
        self.pixel_size = (grid * 16) ** 2 * 3
        if not command or len(profile) != 32 or not callable(admit):
            raise ValueError(
                "worker command, SHA256 profile and admission are required"
            )
        if (
            not math.isfinite(warm_idle_seconds)
            or not 0 <= warm_idle_seconds <= 5
            or (warm_idle_seconds > 0 and (not callable(keep_warm) or not callable(reserve)))
            or max_pending < 1
            or payload_budget_bytes <= 0
            or max_images < 1
            or not 1 <= width <= 32768
            or not math.isfinite(worker_timeout)
            or worker_timeout <= 0
            or not math.isfinite(reserve_wait_timeout)
            or reserve_wait_timeout < 0
        ):
            raise ValueError("invalid vision worker limits")
        self.route = route
        self._queued_images = 0
        self.warm_idle_seconds = warm_idle_seconds
        self.keep_warm = keep_warm
        self._warm_process = None
        self._warm_lease = None
        self._warm_temp = None
        self._warm_errors = None
        self._warm_deadline = 0.0
        self._warm_hits = 0
        self.command = tuple(command)
        self.profile = profile
        self.admit = admit
        self.reserve = reserve
        self.reserve_wait_timeout = reserve_wait_timeout
        self.retry_reserve = retry_reserve
        self.width = width
        self.max_images = max_images
        self.worker_timeout = worker_timeout
        self.payload_budget_bytes = payload_budget_bytes
        self.reserved_bytes = 0
        self.shared_payload_budget = None
        self.slots = threading.BoundedSemaphore(max_pending)
        self.serial = threading.Lock()
        self.last_timings = {}
        self.closed = threading.Event()
        self.metrics_lock = threading.Lock()
        self.metrics = {
            k: 0 for k in ("encoded", "fallback", "cancelled", "failed", "rejected", "worker_starts", "worker_reuses", "worker_stops", "routed_idle", "routed_unavailable", "memory_unavailable")
        }

        if warm_idle_seconds:
            threading.Thread(target=self._reap_idle, daemon=True).start()

    def _count(self, key):
        with self.metrics_lock:
            self.metrics[key] += 1

    def snapshot(self):
        with self.metrics_lock:
            return dict(self.metrics, queued_images=self._queued_images, payload_reserved_bytes=self.reserved_bytes, last_timings=dict(self.last_timings))

    def close(self):
        # Running callers notice within 50 ms, kill/reap only their worker,
        # then release their reservation. No global process-name kills.
        self.closed.set()
        if self.warm_idle_seconds:
            with self.serial:
                self._stop_warm()

    def _check(self, cancelled, deadline):
        if self.closed.is_set():
            raise APIError(503, "server is shutting down", "server_shutdown")
        if cancelled.is_set():
            self._count("cancelled")
            raise APIError(499, "request cancelled", "request_cancelled")
        if time.monotonic() >= deadline:
            raise APIError(504, "request timed out", "request_timeout")

    def _route_to_native(self, request, cancelled, deadline, own_images=0):
        self._check(cancelled, deadline)
        spans = request.image_spans
        if not spans or request.image_embedding_width:
            return True
        # Compiled capacity bounds the padded graph. Unsupported grids retain
        # the native path; admission never changes image pixels or geometry.
        def supported(span):
            h, w = span.grid_height, span.grid_width
            if self.capacity_raw_patches is None:
                return (h, w, span.tokens) == (self.grid, self.grid, self.rows)
            edge = 160 if self.capacity_raw_patches == 14400 else 128
            return (type(h) is int and type(w) is int and 2 <= h <= edge and 2 <= w <= edge
                    and h % 2 == 0 and w % 2 == 0
                    and h * w <= self.capacity_raw_patches and span.tokens == h * w // 4)
        if len(spans) > self.max_images or not all(supported(s) for s in spans):
            self._count("fallback")
            return True
        if len(request.image_pixels) != sum(s.grid_height * s.grid_width * 768 for s in spans):
            raise APIError(400, "invalid image payload", "invalid_image")
        if self.route is not None:
            with self.metrics_lock:
                ahead = max(0, self._queued_images - own_images)
            try:
                decision = self.route(ahead, len(spans))
            except Exception:
                decision = "unavailable"
            self._check(cancelled, deadline)
            if decision != "ane":
                self._count("routed_idle" if decision == "idle" else "routed_unavailable")
                if self.serial.acquire(blocking=False):
                    try:
                        self._stop_warm()
                    finally:
                        self.serial.release()
                return True
        return False

    def prepare(self, request, cancelled, deadline):
        began = time.monotonic()
        self.last_timings = {}
        if self._route_to_native(request, cancelled, deadline):
            return request
        spans = request.image_spans
        if not self.slots.acquire(blocking=False):
            self._count("rejected")
            raise APIError(503, "vision preparation queue is full", "vision_overloaded")
        with self.metrics_lock:
            self._queued_images += len(spans)
        acquired = False
        lease = None
        external = None
        completed = False
        try:
            while not acquired:
                self._check(cancelled, deadline)
                acquired = self.serial.acquire(timeout=0.05)
            self._check(cancelled, deadline)
            # Re-evaluate after waiting for the serial worker slot.
            if self._route_to_native(request, cancelled, deadline, len(spans)):
                return request
            # Failure/absent telemetry fails closed to the governed native path.
            try:
                admitted = self.admit()
            except Exception:
                admitted = False
            if not admitted:
                self._count("fallback")
                return request
            external = self._borrow_warm()
            if self.reserve is not None and external is None:
                waiting_until = min(deadline, time.monotonic() + self.reserve_wait_timeout)
                while True:
                    self._check(cancelled, deadline)
                    try:
                        external = (self.reserve(timeout=min(.5, max(.001, waiting_until - time.monotonic())))
                                    if self.reserve_wait_timeout else self.reserve())
                    except TimeoutError:
                        external = None
                    except Exception:
                        external = None
                        break
                    if external is not None or time.monotonic() >= waiting_until:
                        break
                    if self.retry_reserve is not None:
                        try:
                            retry = self.retry_reserve() is True
                        except Exception:
                            retry = False
                        if not retry:
                            self._count("memory_unavailable")
                            break
                    # Waiting does not occupy a GPU state cell or evict KV.
                    self.closed.wait(min(.05, max(0, waiting_until - time.monotonic())))
                if external is None:
                    self.last_timings["queue_and_admission_seconds"] = time.monotonic() - began
                    self._count("fallback")
                    return request
            self.last_timings["queue_and_admission_seconds"] = time.monotonic() - began
            # A native lease wait can outlast the text prefill. Release the
            # lease through finally rather than start an unhelpful ANE encode.
            if self._route_to_native(request, cancelled, deadline, len(spans)):
                return request
            # Charge the embedding bytes plus serialization/transport copies.
            # This ownership follows the RuntimeCall, not the HTTP connection.
            size = sum(s.tokens for s in spans) * self.width * 2 * 3
            with self.metrics_lock:
                if size > self.payload_budget_bytes - self.reserved_bytes:
                    self.metrics["fallback"] += 1
                    return request
                if self.shared_payload_budget is not None and not self.shared_payload_budget.reserve(size):
                    self.metrics["fallback"] += 1
                    return request
                self.reserved_bytes += size
                lease = _EmbeddingLease(self, size)
            try:
                unique, lookup, placement, grids = [], {}, [], []
                pixel_offset = 0
                for span in spans:
                    pixel_size = span.grid_height * span.grid_width * 768
                    pixels = request.image_pixels[pixel_offset:pixel_offset + pixel_size]
                    pixel_offset += pixel_size
                    key = (span.grid_height, span.grid_width, pixels)
                    index = lookup.get(key)
                    if index is None:
                        index = len(unique)
                        lookup[key] = index
                        unique.append(pixels)
                        grids.append((span.grid_height, span.grid_width))
                    placement.append(index)
                self._active_grids = grids
                self.last_timings["unique_images"] = len(unique)
                self.last_timings["placements"] = len(spans)
                t = time.monotonic()
                payload = self._run(b"".join(unique), len(unique), cancelled, deadline)
                self.last_timings["worker_handoff_validate_seconds"] = time.monotonic() - t
                if len(unique) != len(spans):
                    row_offset, chunks = 0, []
                    for h, w in grids:
                        rows_size = h * w // 4 * self.width * 2
                        chunks.append(payload[row_offset:row_offset + rows_size])
                        row_offset += rows_size
                    payload = b"".join(chunks[i] for i in placement)
            except APIError:
                raise
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
                # _run has killed/reaped the child before fallback can load Metal.
                self._check(cancelled, deadline)
                self._count("failed")
                self._count("fallback")
                return request
            self._check(cancelled, deadline)
            prepared = []
            row_offset = 0
            for span in spans:
                size = span.tokens * self.width * 2
                rows = payload[row_offset:row_offset + size]
                row_offset += size
                digest = hashlib.sha256(
                    b"prepared-vision-bf16-v1\0"
                    + self.profile
                    + struct.pack("<III", span.grid_height, span.grid_width, self.width)
                    + rows
                ).digest()
                lo, hi = struct.unpack("<QQ", digest[:16])
                prepared.append(replace(span, digest_lo=lo, digest_hi=hi))
            self.last_timings["prepare_seconds"] = time.monotonic() - began
            self._count("encoded")
            result = replace(
                request,
                image_owner=(request.image_owner, lease),
                image_spans=tuple(prepared),
                image_pixels=payload,
                image_embedding_width=self.width,
            )
            completed = True
            lease = None  # ownership transferred to result.image_owner
            return result
        finally:
            if acquired:
                self._finish_warm(external, completed)
            elif external is not None:
                external.close()
            if lease is not None:
                lease.close()
            if acquired:
                self.serial.release()
            with self.metrics_lock:
                self._queued_images -= len(spans)
            self.slots.release()

    def _can_keep_warm(self):
        try:
            return not self.closed.is_set() and self.keep_warm() is True
        except Exception:
            return False

    def _stop_warm(self):
        # Caller owns serial. Reservation survives until the process is reaped.
        p = self._warm_process
        if p is not None:
            if p.poll() is None:
                p.kill()
            p.wait()
            self._warm_process = None
            self._count("worker_stops")
        if self._warm_lease is not None:
            self._warm_lease.close()
            self._warm_lease = None
        if self._warm_errors is not None:
            self._warm_errors.close()
            self._warm_errors = None
        if self._warm_temp is not None:
            self._warm_temp.cleanup()
            self._warm_temp = None
        self._warm_hits = 0

    def _borrow_warm(self):
        if self._warm_process is not None:
            if (self._warm_process.poll() is not None
                    or time.monotonic() >= self._warm_deadline
                    or not self._can_keep_warm()):
                self._stop_warm()
        lease, self._warm_lease = self._warm_lease, None
        return lease

    def _finish_warm(self, external, completed):
        if self.warm_idle_seconds:
            if external is not None:
                self._warm_lease = external
            if (completed and self._warm_process is not None
                    and self._warm_process.poll() is None and self._can_keep_warm()):
                # Singletons get only a brief grace period; a real burst earns
                # the configured bounded idle window. Active work is serialized.
                idle = self.warm_idle_seconds if self._warm_hits else min(.25, self.warm_idle_seconds)
                self._warm_deadline = time.monotonic() + idle
            else:
                self._stop_warm()
        elif external is not None:
            external.close()

    def _reap_idle(self):
        while not self.closed.wait(.05):
            if not self.serial.acquire(blocking=False):
                continue
            try:
                if self._warm_process is not None and (
                    self._warm_process.poll() is not None
                    or time.monotonic() >= self._warm_deadline
                    or not self._can_keep_warm()
                ):
                    self._stop_warm()
            finally:
                self.serial.release()

    def _run_warm(self, pixels, count, cancelled, deadline):
        if self._warm_process is None:
            self._warm_temp = tempfile.TemporaryDirectory(prefix="justfit-vision-warm-")
            root = Path(self._warm_temp.name)
            self._warm_errors = (root / "stderr.log").open("wb")
            self._warm_process = subprocess.Popen(
                (*self.command, "--serve", str(root)), stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=self._warm_errors)
            self._count("worker_starts")
        else:
            self._warm_hits += 1
            self._count("worker_reuses")
        root = Path(self._warm_temp.name)
        source, target, done = root / "pixels.rgb", root / "rows.bf16", root / "done.json"
        for path in (target, done):
            path.unlink(missing_ok=True)
        source.write_bytes(pixels)
        request = root / "request.tmp"
        request.write_text(json.dumps({"count": count, "grids": self._active_grids}))
        request.replace(root / "request.json")
        end = min(deadline, time.monotonic() + self.worker_timeout)
        expected = sum(h * w // 4 for h, w in self._active_grids) * self.width * 2
        while not done.exists():
            self._check(cancelled, deadline)
            if self._warm_process.poll() is not None:
                raise RuntimeError("warm worker exited")
            if time.monotonic() >= end:
                raise TimeoutError("warm vision worker timed out")
            if target.exists() and target.stat().st_size > expected:
                raise ValueError("excessive warm worker output")
            cancelled.wait(.01)
        self._check(cancelled, deadline)
        if target.stat().st_size != expected:
            raise ValueError("incorrect warm worker output size")
        payload = target.read_bytes()
        if not _finite_bf16(payload):
            raise ValueError("non-finite warm worker output")
        self.last_timings["worker"] = json.loads(done.read_text())
        return payload

    def prewarm(self, timeout=240.0):
        """Compile and execute once before the native LLM occupies memory.

        Request workers keep their short deadline. Failed startup does not
        advertise an ANE-ready service or repeatedly abort compilation.
        """
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("invalid prewarm deadline")
        with tempfile.TemporaryDirectory(prefix="justfit-vision-prewarm-") as temporary:
            with (Path(temporary) / "stderr.log").open("wb") as errors:
                process = subprocess.Popen(
                    (*self.command, "--prewarm"), stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=errors,
                )
                try:
                    deadline = time.monotonic() + timeout
                    while process.poll() is None:
                        if self.closed.wait(0.05):
                            raise RuntimeError("vision prewarm cancelled")
                        if time.monotonic() >= deadline:
                            raise TimeoutError("vision startup compilation timed out")
                    if process.returncode:
                        errors.flush()
                        detail = (Path(temporary) / "stderr.log").read_bytes()[-4096:]
                        raise RuntimeError("vision prewarm failed: " + detail.decode(errors="replace"))
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.wait()

    def _run(self, pixels, count, cancelled, deadline):
        if self.warm_idle_seconds:
            return self._run_warm(pixels, count, cancelled, deadline)
        expected = sum(h * w // 4 for h, w in self._active_grids) * self.width * 2
        with tempfile.TemporaryDirectory(prefix="justfit-vision-") as temporary:
            root = Path(temporary)
            source, target = root / "pixels.rgb", root / "rows.bf16"
            source.write_bytes(pixels)
            # Files avoid pipe deadlocks and keep protocol stdout out of logs.
            with (root / "stderr.log").open("wb") as errors:
                process = subprocess.Popen(
                    (*self.command, str(source), str(target), str(count),
                     *(("--grids", json.dumps(self._active_grids)) if self.capacity_raw_patches else ())),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=errors,
                )
                try:
                    worker_deadline = min(
                        deadline, time.monotonic() + self.worker_timeout
                    )
                    while process.poll() is None:
                        self._check(cancelled, deadline)
                        if time.monotonic() >= worker_deadline:
                            raise TimeoutError("vision worker timed out")
                        if target.exists() and target.stat().st_size > expected:
                            raise ValueError("vision worker produced excessive output")
                        cancelled.wait(0.05)
                    if process.returncode:
                        raise RuntimeError("vision worker failed")
                    if target.stat().st_size != expected:
                        raise ValueError(
                            "vision worker returned an incorrect row count"
                        )
                    payload = target.read_bytes()
                    t = time.monotonic()
                    finite = _finite_bf16(payload)
                    self.last_timings["validation_seconds"] = time.monotonic() - t
                    detail = (root / "stderr.log").read_text(errors="replace")
                    for line in detail.splitlines():
                        if line.startswith("ANE_WORKER_TIMING "):
                            self.last_timings["worker"] = json.loads(line.split(" ", 1)[1])
                    if not finite:
                        raise ValueError("vision worker returned non-finite rows")
                    return payload
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.wait()


def from_profile(path, runtime, model_root, host_reserve, *, offline_compile=False):
    """Fail closed on deployment/configuration errors; optional at startup only.

    Profile pins every compiled asset/bridge file and original vision weights.
    Variable-grid compilation is offline; startup validates its cache. A
    temporary governor lease protects the ANE worker and is released after
    process exit, including GPU fallback.
    """
    import json

    profile_path = Path(path).resolve(strict=True)
    raw = profile_path.read_bytes()
    profile = json.loads(raw)
    if profile.get("schema") in (2, 3):
        if __package__:
            from .grouped_vision_profile import from_grouped_profile, from_tiered_profile
        else:
            from grouped_vision_profile import from_grouped_profile, from_tiered_profile
        factory = from_grouped_profile if profile["schema"] == 2 else from_tiered_profile
        return factory(profile_path, runtime, model_root, host_reserve)
    if profile.get("schema") != 1 or type(host_reserve) is not int or host_reserve <= 0:
        raise ValueError(
            "ANE vision requires profile schema 1 and explicit positive OS reserve"
        )
    worker_bytes = profile.get("worker_memory_bytes", 1536 * 2**20)
    if type(worker_bytes) is not int or not 512 * 2**20 <= worker_bytes <= 2 * 2**30:
        raise ValueError("ANE worker memory budget must be between 512 MiB and 2 GiB")
    root = profile_path.parent

    def resolve(name):
        p = Path(profile[name])
        return (root / p).resolve(strict=True)

    bridge = resolve("bridge")
    asset = resolve("asset")
    # Preserve the venv executable path: resolving its symlink runs the base
    # interpreter and loses installed NumPy/Core AI dependencies.
    python = (root / Path(profile["python"]).expanduser()).absolute()
    if not python.is_file():
        raise ValueError("ANE worker Python executable does not exist")
    weights = Path(model_root) / "vision/model.bin"

    def digest(p):
        with p.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    if digest(weights) != profile["vision_weights_sha256"]:
        raise ValueError("ANE vision weights do not match the native model package")
    expected = profile["files"]
    actual = {}
    for p in sorted(asset.rglob("*")):
        if p.is_symlink():
            raise ValueError("compiled vision assets cannot contain symlinks")
        if p.is_file():
            actual["asset/" + p.relative_to(asset).as_posix()] = digest(p)
    for name in ("coreai_bridge.py", "libcoreai_bridge.dylib"):
        actual["bridge/" + name] = digest(bridge / name)
    capacity = profile.get("capacity_raw_patches")
    if capacity is not None:
        if type(capacity) is not int or capacity != 4096:
            raise ValueError("unsupported variable-grid profile")
        positions = resolve("position_table")
        actual["position_table"] = digest(positions)
    if not actual or actual != expected:
        raise ValueError("ANE compiled asset or bridge digest mismatch")

    denial_baseline = [None]

    def admit():
        status = json.loads(runtime.status(timeout=0.5).json)
        governor = status.get("memory_governor", {})
        denial_baseline[0] = governor.get("denied_reservations")
        return (
            status.get("ready") is True
            and governor.get("system_pressure") in ("normal", "warning")
            and governor.get("growth_allowed") is True
            and governor.get("host_measurement_valid") is True
            and governor.get("host_reserve_bytes", 0) >= host_reserve
            and governor.get("host_available_bytes", 0) >= host_reserve
            # The atomic safe-point control may release idle model buffers.
            # Current engine headroom alone must not bypass that opportunity.
        )

    grid = profile.get("grid", 32)
    if type(grid) is not int or grid not in (32, 64):
        raise ValueError("unsupported compiled grid")
    entrypoint = profile.get("entrypoint", "vision_tower_ane4_safe")
    if not isinstance(entrypoint, str) or not entrypoint:
        raise ValueError("invalid entrypoint")
    def keep_warm():
        status = json.loads(runtime.status(timeout=0.5).json)
        g = status.get("memory_governor", {})
        scheduler = status.get("scheduler", {})
        admission = status.get("admission", {})
        return (status.get("ready") is True and g.get("system_pressure") == "normal"
                and g.get("growth_allowed") is True and g.get("host_measurement_valid") is True
                and g.get("headroom_bytes", 0) >= 512 * 2**20
                and g.get("host_available_bytes", 0) - max(host_reserve, g.get("host_reserve_bytes", 0)) >= 512 * 2**20
                and not scheduler.get("waiting_resources", 0)
                and not admission.get("waiting_memory", 0))

    policy = profile.get("route_policy", "prefer_ane")
    if policy not in ("prefer_ane", "when_busy", "prefill_overlap"):
        raise ValueError("unsupported vision routing policy")
    if policy == "prefill_overlap":
        if __package__:from .vision_routing import validate_overlap_cost
        else:from vision_routing import validate_overlap_cost
        overlap_cost = validate_overlap_cost(profile.get("route_cost_ms"))
    payload_budget = profile.get("payload_budget_bytes", 256 * 2**20 if grid == 64 else 64 * 2**20)
    if type(payload_budget) is not int or not 1 <= payload_budget <= 512 * 2**20:
        raise ValueError("invalid vision payload budget")

    def route(ahead, images):
        if policy == "prefer_ane":
            return "ane"
        status = json.loads(runtime.status(timeout=0.5).json)
        if __package__:
            from .vision_routing import choose_route
        else:
            from vision_routing import choose_route
        if policy == "prefill_overlap":
            if __package__:from .vision_routing import choose_overlap_route
            else:from vision_routing import choose_overlap_route
            return choose_overlap_route(status, ahead, images, overlap_cost)
        return choose_route(status, ahead)

    def retry_reserve():
        if __package__:
            from .vision_routing import can_retry_reservation
        else:
            from vision_routing import can_retry_reservation
        status = json.loads(runtime.status(timeout=0.5).json)
        denials = status.get("memory_governor", {}).get("denied_reservations")
        # A timed-out control or busy GPU vision may not have reached native
        # reclamation yet. Do not mistake that for a confirmed budget denial.
        if type(denials) is int and type(denial_baseline[0]) is int and denials <= denial_baseline[0]:
            return True
        return can_retry_reservation(status, worker_bytes)

    worker = PreparedVision(
        [
            str(python),
            str(Path(__file__).with_name("ane_vision_worker.py")),
            str(bridge),
            str(asset),
            "--grid", str(grid), "--entrypoint", entrypoint,
            *(("--capacity", str(capacity), "--positions", str(positions),
               "--cache-record", str(root / profile["cache_record"])) if capacity else ()),
            *(("--cached-only",) if capacity and not offline_compile else ()),
        ],
        hashlib.sha256(raw).digest(),
        admit,
        reserve=lambda **kw: runtime.reserve_external_memory(worker_bytes, **kw),
        reserve_wait_timeout=30.0,
        retry_reserve=retry_reserve,
        grid=grid,
        warm_idle_seconds=profile.get("warm_idle_seconds", 0.0),
        keep_warm=keep_warm,
        route=route,
        payload_budget_bytes=payload_budget,
        capacity_raw_patches=capacity,
    )
    try:
        startup_timeout = profile.get("startup_timeout_seconds", 600.0)
        if (isinstance(startup_timeout, bool) or not isinstance(startup_timeout, (int, float))
                or not math.isfinite(startup_timeout) or not 1 <= startup_timeout <= 1800):
            raise ValueError("invalid vision startup timeout")
        lease = runtime.reserve_external_memory(worker_bytes, timeout=30)
        if lease is None:
            raise RuntimeError("ANE vision startup lease denied")
        try:
            worker.prewarm(timeout=startup_timeout)
        finally:
            lease.close()
    except BaseException:
        worker.close()
        raise
    return worker
