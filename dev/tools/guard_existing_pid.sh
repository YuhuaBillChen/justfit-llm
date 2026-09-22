#!/usr/bin/env bash
# Sample one existing process's macOS phys_footprint and terminate it at a
# fixed threshold. Intended only for bounded qualification runs.
set -u

if [[ $# -ne 3 ]]; then
  echo "usage: $0 PID THRESHOLD_MIB LOG" >&2
  exit 2
fi
pid="$1"
threshold_mib="$2"
log="$3"
threshold_bytes=$((threshold_mib * 1024 * 1024))

printf '%s guard pid=%s threshold=%sMiB\n' \
  "$(date +%H:%M:%S)" "$pid" "$threshold_mib" >>"$log"
while kill -0 "$pid" 2>/dev/null; do
  bytes="$(/usr/bin/footprint --pid "$pid" --noCategories --format bytes \
    2>/dev/null | awk '/phys_footprint:/ {print $2; exit}')"
  if [[ "$bytes" =~ ^[0-9]+$ ]]; then
    printf '%s pid=%s footprint_bytes=%s footprint_mib=%d\n' \
      "$(date +%H:%M:%S)" "$pid" "$bytes" "$((bytes / 1024 / 1024))" \
      >>"$log"
    if ((bytes >= threshold_bytes)); then
      printf '%s FOOTPRINT_GUARD_TRIPPED pid=%s footprint_bytes=%s threshold_bytes=%s\n' \
        "$(date +%H:%M:%S)" "$pid" "$bytes" "$threshold_bytes" >>"$log"
      kill -TERM "$pid" 2>/dev/null || true
      exit 42
    fi
  fi
  sleep 0.25
done
printf '%s guard process-exited pid=%s\n' "$(date +%H:%M:%S)" "$pid" >>"$log"
