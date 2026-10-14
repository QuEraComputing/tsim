#!/bin/zsh
# Run the whole Tsim/Clifft comparison on an idle machine and write report.md.
#
# Run from the repository root:
#
#     scripts/clifft_comparison/run_all.sh [OUT_DIR]
#
# Before each step this waits until the summed %CPU of all processes stays below
# CPU_MAX (default 300, three cores) for three checks a minute apart. (macOS's
# load average counts waiting threads, so it sits near 5 on an idle desktop.)
# load.log records CPU use and load every 30 s throughout, so interference shows
# up afterwards. SECONDS_PER (default 10) is the timed window of every
# measurement.

set -euo pipefail
CPU_MAX=${CPU_MAX:-300}
SECONDS_PER=${SECONDS_PER:-10}
CLIFFT_PYTHON=${CLIFFT_PYTHON:-/tmp/clifft-venv/bin/python}
OUT=${1:-"$(git rev-parse --path-format=absolute --git-common-dir)/../../clifft-bench-results/statevector-vs-clifft-$(date +%Y%m%d-%H%M%S)"}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
HERE=${0:A:h}

load() { sysctl -n vm.loadavg | awk '{print $2}'; }
cpu() { ps -Ao pcpu= | awk '{s += $1} END {printf "%.0f", s}'; }

wait_for_idle() {
  local calm=0
  while (( calm < 3 )); do
    if (( $(cpu) < CPU_MAX )); then calm=$((calm + 1)); else calm=0; fi
    echo "$(date '+%F %T') waiting for idle before $1: cpu $(cpu)%, load $(load), calm $calm/3" >> "$OUT/progress.log"
    (( calm < 3 )) && sleep 60
  done
  echo "$(date '+%F %T') start $1 (cpu $(cpu)%, load $(load))" >> "$OUT/progress.log"
  { ps -Ao pcpu,command -r | head -6 | cut -c1-160 >> "$OUT/progress.log"; } || true
}

(while true; do echo "$(date '+%F %T') cpu $(cpu)% load $(load)" >> "$OUT/load.log"; sleep 30; done) &
LOGGER=$!
trap 'kill $LOGGER 2>/dev/null' EXIT

{
  echo "tsim commit: $(git rev-parse HEAD) (dirty: $([[ -n $(git status --porcelain) ]] && echo yes || echo no))"
  echo "clifft: $($CLIFFT_PYTHON -c 'import clifft; print(clifft.__version__)')"
  echo "jax: $(uv run python -c 'import jax; print(jax.__version__)' 2>/dev/null)"
  echo "machine: $(sysctl -n machdep.cpu.brand_string), $(sysctl -n hw.ncpu) cores, $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"
  echo "macOS: $(sw_vers -productVersion)"
  echo "seconds per measurement: $SECONDS_PER"
} > "$OUT/environment.txt"

wait_for_idle clifft
caffeinate -i "$CLIFFT_PYTHON" "$HERE/run_clifft.py" --seconds "$SECONDS_PER" \
  --out "$OUT/clifft.jsonl" > "$OUT/clifft.log" 2>&1
wait_for_idle "tsim, all threads"
caffeinate -i uv run python "$HERE/run_tsim.py" --seconds "$SECONDS_PER" \
  --out "$OUT/tsim-all.jsonl" > "$OUT/tsim-all.log" 2>&1
wait_for_idle "tsim, 1 thread"
NPROC=1 NUMBA_NUM_THREADS=1 caffeinate -i uv run python "$HERE/run_tsim.py" \
  --seconds "$SECONDS_PER" \
  --out "$OUT/tsim-1.jsonl" > "$OUT/tsim-1.log" 2>&1

uv run python "$HERE/report.py" "$OUT" > "$OUT/report.md"
echo "$(date '+%F %T') done" >> "$OUT/progress.log"
