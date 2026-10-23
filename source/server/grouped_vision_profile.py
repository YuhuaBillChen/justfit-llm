"""Experimental cached-only tier adapter; activation requires explicit profiles."""
from __future__ import annotations
import hashlib,json,math,threading,sys
from pathlib import Path
if __package__:
    from .prepared_vision import PreparedVision,PayloadBudget
    from .vision_routing import choose_route,can_retry_reservation,choose_overlap_route,validate_overlap_cost,choose_relief_route,validate_relief_phases
else:
    from prepared_vision import PreparedVision,PayloadBudget
    from vision_routing import choose_route,can_retry_reservation,choose_overlap_route,validate_overlap_cost,choose_relief_route,validate_relief_phases

class TieredPreparedVision:
    def __init__(self, workers, gpu_only_capacities=()):
        allowed=(4096,8192,12288,14400,16384)
        if (not isinstance(gpu_only_capacities,(tuple,list))
                or any(type(k) is not int or k not in allowed for k in gpu_only_capacities)
                or len(set(gpu_only_capacities))!=len(gpu_only_capacities)
                or set(gpu_only_capacities)&set(workers)):
            raise ValueError('invalid or ambiguous GPU-only tier')
        self.gpu_only_capacities=tuple(sorted(gpu_only_capacities))
        if (not workers and not self.gpu_only_capacities) or any(k not in allowed for k in workers):
            raise ValueError('unsupported/empty vision tier collection')
        self.workers=dict(sorted(workers.items()))
        self.payload_budget=PayloadBudget(512*2**20)
        for worker in self.workers.values():worker.shared_payload_budget=self.payload_budget
        # Prevent simultaneous per-tier leases and NPU encodes. Large tiers
        # retain no idle process; a small warm tier is reaped before switching.
        self.serial=threading.Lock()
        self.last_timings={}

    def prepare(self, request, cancelled, deadline):
        spans=request.image_spans
        if not spans or request.image_embedding_width:return request
        capacities=sorted(set(self.workers)|set(self.gpu_only_capacities))
        for span in spans:
            required=span.grid_height*span.grid_width
            smallest=next((n for n in capacities if n>=required),None)
            if smallest in self.gpu_only_capacities:return request
        capacity=max(s.grid_height*s.grid_width for s in spans)
        selected=next((w for n,w in self.workers.items() if n>=capacity),None)
        if selected is None:return request
        # Native-only routing needs no NPU lock. Recheck inside prepare after
        # acquiring the lock when the first decision actually selects ANE.
        if selected._route_to_native(request,cancelled,deadline):
            print('ANE_VISION_EVENT '+json.dumps(dict(capacity=selected.capacity_raw_patches,prepared=False,native_bypass=True,metrics=selected.snapshot())),file=sys.stderr,flush=True)
            return request
        while not self.serial.acquire(timeout=.05):selected._check(cancelled,deadline)
        try:
            for worker in self.workers.values():
                if worker is not selected and worker.warm_idle_seconds:
                    with worker.serial:worker._stop_warm()
            result=selected.prepare(request,cancelled,deadline)
            self.last_timings=dict(selected.last_timings,tier_capacity=selected.capacity_raw_patches)
            print('ANE_VISION_EVENT '+json.dumps(dict(capacity=selected.capacity_raw_patches,prepared=result.image_embedding_width!=0,metrics=selected.snapshot())),file=sys.stderr,flush=True)
            return result
        finally:self.serial.release()

    def close(self):
        for worker in self.workers.values():worker.close()

    def snapshot(self):return dict({str(n):w.snapshot() for n,w in self.workers.items()},shared_payload=self.payload_budget.snapshot(),gpu_only_capacities=list(self.gpu_only_capacities))

def from_grouped_profile(path,runtime,model_root,host_reserve):
    """Validate immutable files and cached placement; never offline-compile here."""
    if type(host_reserve) is not int or host_reserve<=0:raise ValueError('positive explicit host reserve required')
    path=Path(path).resolve(strict=True);raw=path.read_bytes();profile=json.loads(raw);root=path.parent
    if profile.get('schema')!=2 or profile.get('kind')!='grouped-vision':raise ValueError('unsupported grouped profile')
    def resolve(key):return (root/Path(profile[key])).resolve(strict=True)
    def digest(p):
        with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    if digest(Path(model_root)/'vision/model.bin')!=profile['vision_weights_sha256']:raise ValueError('vision weights mismatch')
    capacity=profile.get('capacity_raw_patches')
    if type(capacity) is not int or capacity not in (4096,8192,12288,14400,16384):raise ValueError('unsupported grouped capacity')
    worker_bytes=profile.get('worker_memory_bytes')
    if type(worker_bytes) is not int or not 512*2**20<=worker_bytes<=1127*2**20:raise ValueError('grouped worker exceeds fixed budget')
    timeout=profile.get('worker_timeout_seconds')
    if isinstance(timeout,bool) or not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or not 10<=timeout<=600:raise ValueError('invalid grouped timeout')
    # Each entry is a literal profile-relative path, not a traversable asset glob.
    files=profile.get('files')
    if not isinstance(files,dict) or not files:raise ValueError('empty grouped file pins')
    for name,sha in files.items():
        p=root/Path(name)
        if p.is_symlink() or digest(p)!=sha:raise ValueError('grouped file digest mismatch')
    assets=resolve('assets');positions=resolve('position_table');catalog=resolve('cache_record');script=resolve('worker')
    required={positions,catalog,script}
    required.update((script.parent/name).resolve(strict=True) for name in ('warm_chain.py','graph.py','partition.py'))
    count=profile.get('group_count',9)
    if type(count) is not int or count not in (9,27):raise ValueError('unsupported vision group count')
    partition=assets/'partition.json'
    if count==27:
        row=json.loads(partition.read_text())
        if capacity not in (12288,14400) or row.get('capacity')!=capacity or row.get('groups')!=[[i,i+1] for i in range(27)]:raise ValueError('incomplete vision block partition')
        required.add(partition.resolve(strict=True))
    elif partition.exists():raise ValueError('profile group count disagrees with partition')
    for i in range(count):
        asset=assets/f'group-{i:02d}'/'model.aimodel'
        if not asset.is_dir() or asset.is_symlink():raise ValueError('missing or linked group asset')
        for item in asset.rglob('*'):
            if item.is_symlink():raise ValueError('linked group asset file')
            if item.is_file():required.add(item.resolve(strict=True))
    pinned={(root/Path(name)).resolve(strict=True) for name in files}
    if not required<=pinned:raise ValueError('grouped asset/worker/catalog file pins incomplete')
    # Preserve venv symlink, as resolving it selects the base interpreter.
    python=(root/Path(profile['python']).expanduser()).absolute()
    if not python.is_file():raise ValueError('worker interpreter absent')
    policy=profile.get('route_policy','prefill_overlap')
    if policy not in ('prefill_overlap','gpu_relief'):raise ValueError('unsupported grouped vision routing policy')
    cost=validate_overlap_cost(profile.get('route_cost_ms')) if policy=='prefill_overlap' else None
    phases=validate_relief_phases(profile.get('phase_qualification')) if policy=='gpu_relief' else None
    def route(ahead,images):
        current=status()
        if phases is not None:return choose_relief_route(current,ahead,images,phases)
        return choose_overlap_route(current,ahead,images,cost)
    def status():return json.loads(runtime.status(timeout=.5).json)
    def admit():
        s=status();g=s.get('memory_governor',{})
        return (s.get('ready') is True and g.get('system_pressure') in ('normal','warning')
                and g.get('growth_allowed') is True and g.get('host_measurement_valid') is True
                and g.get('host_reserve_bytes',0)>=host_reserve and g.get('host_available_bytes',0)>=host_reserve)
    worker=PreparedVision([str(python),str(script),'--assets',str(assets),'--positions',str(positions),
                           '--capacity',str(capacity),'--catalog',str(catalog)],hashlib.sha256(raw).digest(),admit,
                          reserve=lambda **kw:runtime.reserve_external_memory(worker_bytes,**kw),
                          reserve_wait_timeout=30,retry_reserve=lambda:can_retry_reservation(status(),worker_bytes),
                          route=route,capacity_raw_patches=capacity,
                          worker_timeout=timeout,payload_budget_bytes=512*2**20,warm_idle_seconds=0)
    # Cached startup loading is also an external worker, before requests.
    lease=runtime.reserve_external_memory(worker_bytes,timeout=30)
    if lease is None:
        if policy=='gpu_relief':
            # No child or lease exists. Requests retain normal admission and
            # cached-only placement validation; denial safely falls back to GPU.
            worker._count('memory_unavailable')
            return worker
        worker.close()
        raise RuntimeError('ANE vision startup lease denied')
    try:worker.prewarm(timeout=profile.get('startup_timeout_seconds',120))
    except BaseException:worker.close();raise
    finally:lease.close()  # prewarm always kills/reaps its child before returning
    return worker

def from_tiered_profile(path,runtime,model_root,host_reserve):
    if __package__:from .prepared_vision import from_profile
    else:from prepared_vision import from_profile
    path=Path(path).resolve(strict=True);data=json.loads(path.read_text());workers={}
    if data.get('schema')!=3 or data.get('kind')!='tiered-vision' or not isinstance(data.get('tiers'),list) or not 0<=len(data['tiers'])<=4:raise ValueError('invalid tiered profile')
    try:
        for tier in data['tiers']:
            child=(path.parent/Path(tier['profile'])).resolve(strict=True);raw=child.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=tier['sha256']:raise ValueError('tier profile digest mismatch')
            meta=json.loads(raw)
            if meta.get('schema') not in (1,2):raise ValueError('tier profile must be a leaf')
            capacity=meta.get('capacity_raw_patches')
            if capacity not in (4096,8192,12288,14400,16384) or capacity in workers:raise ValueError('duplicate/unsupported tier')
            workers[capacity]=from_profile(child,runtime,model_root,host_reserve)
        return TieredPreparedVision(workers,data.get('gpu_only_capacities',()))
    except BaseException:
        for worker in workers.values():worker.close()
        raise
