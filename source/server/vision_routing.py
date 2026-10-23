"""Select the vision device without overflowing a busy accelerator queue."""

def choose_route(status, images_ahead=0):
    if status.get('ready') is not True:
        return 'unavailable'
    scheduler = status.get('scheduler')
    if not isinstance(scheduler, dict):
        return 'unavailable'
    counters = [scheduler.get(key) for key in ('prefilling', 'decoding', 'queued')]
    if any(type(value) is not int or value < 0 for value in counters):
        return 'unavailable'
    if images_ahead > 0 or any(counters):
        return 'ane'
    return 'idle'


def can_retry_reservation(status, required_bytes):
    """After a native denial, wait only for a plausible resource transition.

    The native safe-point attempt always comes first so idle vision/state
    buffers get a chance to release. This bounds waiting, not memory admission;
    every retry still needs a real native lease.
    """
    try:
        governor = status['memory_governor']
        if status.get('ready') is not True or governor['system_pressure'] == 'critical':
            return False
        # 2.3 snapshot: limit - charged - positive headroom is the exact
        # reservation total. No absent phase/state sizing earns extra credit.
        reserved = governor.get('reserved_bytes')
        if reserved is None:
            limit, charged, headroom = (governor[k] for k in
                                       ('limit_bytes','charged_bytes','headroom_bytes'))
            if any(type(v) is not int or v < 0 for v in (limit,charged,headroom)):
                return False
            reserved = max(0, limit - charged - headroom)
        if type(reserved) is not int or reserved < 0:
            return False
        if reserved > 0:
            return True
        available = governor['headroom_bytes']
        if type(available) is not int or available < 0:
            return False
        phase = status.get('phase_residency', {})
        memory = status.get('memory_plan', {}).get('model', {}).get('memory', {})
        if phase.get('prefill_arena_resident') is True:
            available += memory.get('shared_prefill_bytes', 0)
        kv = status.get('kv', {})
        available += kv.get('reclaimable_backing_bytes', kv.get('reclaimable_bytes', 0))
        available += status.get('state', {}).get('warm_idle_cells', 0) * memory.get('active_state_cell_bytes', 0)
        return available >= required_bytes
    except (KeyError, TypeError):
        return False


def validate_overlap_cost(cost):
    """Costs are profile-pinned measurements for this hardware and image tier.

    ANE time includes worker startup, cached loads and output transfer. GPU time
    excludes waiting behind text. No online compilation or invented token cutoff.
    """
    import math
    if not isinstance(cost, dict):
        raise ValueError('prefill_overlap requires measured route_cost_ms')
    for key in ('ane_image', 'gpu_image', 'minimum_gain'):
        value = cost.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('invalid overlap cost: ' + key)
    penalty = cost.get('text_penalty')
    if type(penalty) not in (int, float) or not math.isfinite(penalty) or penalty < 0:
        raise ValueError('measured text contention penalty required')
    if cost.get('concurrent_qualified') is not True:
        raise ValueError('isolated encoder timing does not qualify overlap')
    return dict(cost)


def choose_overlap_route(status, images_ahead, images, cost):
    """Offload only when another runnable text prefill can hide ANE latency."""
    import math
    try:
        cost = validate_overlap_cost(cost)
        if status.get('ready') is not True:
            return 'unavailable'
        scheduler = status['scheduler']
        if type(images_ahead) is not int or images_ahead != 0:
            return 'idle'  # Do not create a second unmeasured ANE queue.
        if type(images) is not int or images < 1:
            return 'unavailable'
        active = scheduler['prefilling']
        if type(active) is not int or active < 0:
            return 'unavailable'
        if not active:
            return 'idle'  # Decode-only and merely queued work do not qualify.
        if scheduler.get('prefill_estimate_valid') is not True:
            return 'unavailable'
        remaining = scheduler['runnable_prefill_remaining_ms']
        if type(remaining) not in (int, float) or not math.isfinite(remaining) or remaining < 0:
            return 'unavailable'
        # Compare whole completion time, including measured shared-memory
        # contention. An idle-encoder win alone cannot justify offloading.
        gpu_finish = remaining + images * cost['gpu_image']
        ane_finish = max(remaining + cost['text_penalty'], images * cost['ane_image'])
        return 'ane' if gpu_finish - ane_finish >= cost['minimum_gain'] else 'idle'
    except (KeyError, TypeError, ValueError):
        return 'unavailable'


def validate_relief_phases(phases):
    """Explicit per-node/tier acceptance; no borrowed makespan cost cutoff."""
    if (not isinstance(phases, dict) or set(phases) != {'prefill', 'decode'}
            or any(type(value) is not bool for value in phases.values())):
        raise ValueError('gpu_relief requires boolean prefill/decode qualification')
    return dict(phases)


def choose_relief_route(status, images_ahead, images, phases):
    """Free a busy GPU while preserving language ANE prefill priority.

    This is an eligibility check, repeated after lease waits. Native vision
    ownership remains the authority if a new text prefill starts afterwards:
    FFN split is suspended until the external vision lease is released.
    """
    try:
        phases = validate_relief_phases(phases)
        if status.get('ready') is not True:
            return 'unavailable'
        scheduler = status['scheduler']
        counters = [scheduler[key] for key in ('prefilling', 'decoding', 'queued')]
        if any(type(value) is not int or value < 0 for value in counters):
            return 'unavailable'
        if type(images_ahead) is not int or images_ahead < 0 or type(images) is not int or images < 1:
            return 'unavailable'
        if images_ahead or images != 1:
            return 'idle'
        ffn = status['ane_ffn']['state']
        if ffn not in ('off', 'split', 'stopped'):
            return 'unavailable'
        prefilling, decoding, _ = counters
        if prefilling:
            return 'ane' if ffn == 'off' and phases['prefill'] else 'idle'
        if decoding and phases['decode']:
            return 'ane'
        return 'idle'
    except (KeyError, TypeError, ValueError):
        return 'unavailable'
