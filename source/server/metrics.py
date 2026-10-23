# Modified for JustFit: product identity and integration.
"""Usage, metrics and Prometheus text rendering for request results."""

from __future__ import annotations

import math

from .latency import prometheus_latency


def is_finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError):
        return False


# Each gauge or counter Prometheus reads from the status, by the path of
# keys to its value.
PROMETHEUS_SERIES = {
    "justfit_ready": ("ready",),
    "justfit_metal_healthy": ("metal", "healthy"),
    "justfit_transport_pending": ("transport", "pending"),
    "justfit_frontend_active": ("frontend", "active"),
    "justfit_frontend_waiting": ("frontend", "waiting"),
    "justfit_requests_submitted_total": ("requests", "submitted"),
    "justfit_requests_completed_total": ("requests", "completed"),
    "justfit_requests_cancelled_total": ("requests", "cancelled"),
    "justfit_requests_failed_total": ("requests", "failed"),
    "justfit_scheduler_queued": ("scheduler", "queued"),
    "justfit_scheduler_waiting_resources": ("scheduler", "waiting_resources"),
    "justfit_scheduler_waiting_prefix": ("scheduler", "waiting_prefix"),
    "justfit_cache_resource_suspensions_total": ("cache", "resource_suspensions"),
    "justfit_cache_priority_suspensions_total": ("cache", "priority_suspensions"),
    "justfit_cache_resource_resumptions_total": ("cache", "resource_resumptions"),
    "justfit_cache_resource_replay_tokens_total": (
        "cache",
        "resource_replay_tokens",
    ),
    "justfit_scheduler_prefilling": ("scheduler", "prefilling"),
    "justfit_scheduler_decoding": ("scheduler", "decoding"),
    "justfit_scheduler_waiting_mask": ("scheduler", "waiting_mask"),
    "justfit_scheduler_prefill_batches_total": ("scheduler", "prefill_batches"),
    "justfit_scheduler_prefill_rows_total": ("scheduler", "prefill_rows"),
    "justfit_scheduler_decode_batches_total": ("scheduler", "decode_batches"),
    "justfit_scheduler_decode_b1_total": (
        "scheduler",
        "decode_batches_by_width",
        "b1",
    ),
    "justfit_scheduler_decode_b2_total": (
        "scheduler",
        "decode_batches_by_width",
        "b2",
    ),
    "justfit_scheduler_decode_b3_total": (
        "scheduler",
        "decode_batches_by_width",
        "b3",
    ),
    "justfit_scheduler_decode_b4_total": (
        "scheduler",
        "decode_batches_by_width",
        "b4",
    ),
    "justfit_kv_pages_allocated": ("kv", "pages_allocated"),
    "justfit_kv_pages_active": ("kv", "pages_active"),
    "justfit_kv_pages_cache": ("kv", "pages_cache"),
    "justfit_kv_free_allocated_pages": ("kv", "pages_free"),
    "justfit_kv_allocated_bytes": ("kv", "allocated_bytes"),
    "justfit_kv_reclaimable_bytes": ("kv", "reclaimable_bytes"),
    "justfit_kv_extent_allocations_total": ("kv", "extent_allocations"),
    "justfit_kv_extent_releases_total": ("kv", "extent_releases"),
    "justfit_kv_extent_allocate_max_milliseconds": (
        "kv",
        "extent_allocate_max_ms",
    ),
    "justfit_kv_extent_release_max_milliseconds": (
        "kv",
        "extent_release_max_ms",
    ),
    "justfit_kv_extent_compactions_total": ("kv", "extent_compactions"),
    "justfit_kv_pages_moved_total": ("kv", "pages_moved"),
    "justfit_kv_extent_compact_max_milliseconds": (
        "kv",
        "extent_compact_max_ms",
    ),
    "justfit_state_entries": ("state", "entries"),
    "justfit_state_pinned": ("state", "pinned"),
    "justfit_state_in_use": ("state", "in_use"),
    "justfit_state_in_use_evictions_total": ("state", "in_use_evictions"),
    "justfit_state_bytes": ("state", "bytes"),
    "justfit_state_active_lanes": ("state", "active_lanes"),
    "justfit_state_publications_total": ("state", "publications"),
    "justfit_state_evictions_total": ("state", "evictions"),
    "justfit_cache_hits_total": ("cache", "hits"),
    "justfit_cache_cold_misses_total": ("cache", "cold_misses"),
    "justfit_cache_reused_tokens_total": ("cache", "reused_tokens"),
    "justfit_cache_lazy_junctions_total": ("cache", "lazy_junctions"),
    "justfit_target_prefill_rows_total": (
        "draft_context",
        "target_prefill_rows",
    ),
    "justfit_draft_context_prompt_end_rows_total": (
        "draft_context",
        "prompt_end_rows",
    ),
    "justfit_draft_context_materialization_rows_total": (
        "draft_context",
        "materialization_rows",
    ),
    "justfit_draft_context_avoided_rows_total": (
        "draft_context",
        "avoided_rows",
    ),
    "justfit_draft_state_restore_skipped_total": (
        "draft_context",
        "restore_skipped",
    ),
    "justfit_draft_state_resets_total": ("draft_context", "resets"),
    "justfit_constraint_mask_overlap_batches_total": (
        "constraint_masks",
        "overlap_batches",
    ),
    "justfit_constraint_mask_overlap_requests_total": (
        "constraint_masks",
        "overlap_requests",
    ),
    "justfit_constraint_mask_target_forward_gpu_milliseconds": (
        "constraint_masks",
        "last_target_forward_gpu_ms",
    ),
    "justfit_constraint_mask_residual_wait_milliseconds": (
        "constraint_masks",
        "last_residual_wait_ms",
    ),
    "justfit_image_encodes_total": ("images", "encodes"),
    "justfit_image_embedding_reuses_total": ("images", "embedding_reuses"),
    "justfit_image_arena_bytes": ("images", "arena_bytes"),
    "justfit_image_cached_bytes": ("images", "cached_bytes"),
    "justfit_memory_current_bytes": ("memory_actual", "current_bytes"),
    "justfit_memory_peak_bytes": ("memory_actual", "peak_bytes"),
    "justfit_memory_denied_reservations_total": (
        "memory_governor",
        "denied_reservations",
    ),
    "justfit_memory_limit_bytes": ("memory_governor", "limit_bytes"),
    "justfit_memory_headroom_bytes": ("memory_governor", "headroom_bytes"),
    "justfit_admission_waiting_memory": ("admission", "waiting_memory"),
    "justfit_admission_waiting_concurrency": ("admission", "waiting_concurrency"),
    "justfit_admission_held_behind_refusal": ("admission", "held_behind_refusal"),
    "justfit_admission_restoring": ("admission", "restoring"),
    "justfit_admission_suspended": ("admission", "suspended"),
    "justfit_admission_oldest_wait_milliseconds": ("admission", "oldest_wait_ms"),
    "justfit_ttft_p50_milliseconds": ("metrics", "ttft_ms", "p50"),
    "justfit_ttft_p95_milliseconds": ("metrics", "ttft_ms", "p95"),
    "justfit_itl_p50_milliseconds": ("metrics", "itl_ms", "p50"),
    "justfit_itl_p95_milliseconds": ("metrics", "itl_ms", "p95"),
    "justfit_prefill_input_tokens_total": ("metrics", "prefill_input_tokens"),
    "justfit_prefill_wall_milliseconds_total": ("metrics", "prefill_wall_ms"),
    "justfit_prefill_tokens_per_second": (
        "metrics",
        "prefill_tokens_per_second",
    ),
    "justfit_decode_output_tokens_total": ("metrics", "decode_output_tokens"),
    "justfit_decode_wall_milliseconds_total": ("metrics", "decode_wall_ms"),
    "justfit_decode_cycle_milliseconds_total": ("metrics", "decode_cycle_ms"),
    "justfit_decode_tokens_per_second": (
        "metrics",
        "decode_tokens_per_second",
    ),
    "justfit_drafted_tokens_total": ("metrics", "drafted_tokens"),
    "justfit_accepted_draft_tokens_total": ("metrics", "accepted_draft_tokens"),
    "justfit_draft_acceptance_ratio": ("metrics", "draft_acceptance_rate"),
    "justfit_capacity_failures_total": ("metrics", "capacity_failures"),
    "justfit_metal_failures_total": ("metrics", "metal_failures"),
    "justfit_response_store_entries": ("response_store", "entries"),
    "justfit_response_store_bytes": ("response_store", "bytes"),
}


def prometheus_metrics(status):
    """Render low-cardinality metrics directly from the native status."""

    def value(path):
        current = status
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        if isinstance(current, bool):
            return 1 if current else 0
        if is_finite_number(current):
            return current
        return None

    lines = [
        "# HELP justfit_info JustFit runtime metrics.",
        "# TYPE justfit_info gauge",
        'justfit_info{runtime="native"} 1',
    ]
    pressure = status.get("memory_pressure")
    for state in ("normal", "warning", "critical"):
        lines.append(
            f'justfit_memory_pressure{{state="{state}"}} {1 if pressure == state else 0}'
        )
    for name, path in PROMETHEUS_SERIES.items():
        metric_value = value(path)
        if metric_value is not None:
            lines.append(f"{name} {metric_value}")
    lines.extend(prometheus_latency(status.get("latency", {})))
    return "\n".join(lines) + "\n"


def usage_dict(result, job):
    return {
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.prompt_tokens + result.completion_tokens,
        "prompt_tokens_details": {"cached_tokens": result.cache.matched_tokens},
        "completion_tokens_details": {"reasoning_tokens": job.reasoning_tokens},
    }


def timings_dict(result):
    """llama-server-compatible counts and request lifecycle timings.

    Counts are final totals. Rates exclude cached prompt tokens and the first
    emission respectively: those tokens precede the intervals being measured.
    These are elapsed request intervals, not isolated GPU execution times.
    """
    latency = result.metrics["request_latency"]
    prompt_ms = latency.get("start_to_first_token_ms", 0.0)
    prompt_rate = result.prefill_tokens * 1000.0 / prompt_ms if prompt_ms else 0.0
    return {
        "prompt_n": result.prompt_tokens,
        "prompt_ms": prompt_ms,
        "prompt_per_second": prompt_rate if math.isfinite(prompt_rate) else 0.0,
        "predicted_n": result.completion_tokens,
        "predicted_ms": latency.get("first_token_to_done_ms", 0.0),
        "predicted_per_second": latency.get("stream_tokens_per_second", 0.0),
        "cache_n": result.cache.matched_tokens,
    }


def metrics_dict(result):
    # The first native emission can contain a whole speculative block. Its
    # generation precedes TTFT, so none of those tokens belong in the interval
    # from first emission to Done. Native command throughput lives in /status.
    decode_tokens = max(0, result.completion_tokens - result.first_token_batch_tokens)
    latency = {}
    intervals = (
        ("start_to_first_token_ms", result.start_to_first_token_ms),
        ("first_token_to_done_ms", result.first_token_to_done_ms),
        ("wall_ms", result.request_wall_ms),
    )
    for name, milliseconds in intervals:
        if math.isfinite(milliseconds) and milliseconds >= 0:
            latency[name] = milliseconds
    if (
        result.completion_tokens > 0
        and "wall_ms" in latency
        and "first_token_to_done_ms" in latency
        and latency["wall_ms"] >= latency["first_token_to_done_ms"]
    ):
        latency["ttft_ms"] = latency["wall_ms"] - latency["first_token_to_done_ms"]
        if (
            "start_to_first_token_ms" in latency
            and latency["ttft_ms"] >= latency["start_to_first_token_ms"]
        ):
            latency["queue_to_start_ms"] = (
                latency["ttft_ms"] - latency["start_to_first_token_ms"]
            )
    if (
        result.first_token_batch_tokens > 0
        and decode_tokens > 0
        and latency.get("first_token_to_done_ms", 0) > 0
    ):
        rate = decode_tokens * 1000.0 / latency["first_token_to_done_ms"]
        if math.isfinite(rate):
            latency["stream_tokens_per_second"] = rate
    cache_info = result.cache
    cache = {
        "status": cache_info.status,
        "matched_tokens": cache_info.matched_tokens,
        "lane": cache_info.lane if cache_info.lane >= 0 else None,
    }
    metrics = {
        "prefill": {"tokens": result.prefill_tokens},
        "decode": {"tokens": decode_tokens},
        # Native DoneEvent intervals are request lifecycle timing, not summed
        # executor command/GPU time. Native batch throughput lives in /status.
        "request_latency": latency,
        "cache": cache,
    }
    return metrics
