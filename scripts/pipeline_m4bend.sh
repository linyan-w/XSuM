#!/usr/bin/env bash
# M4 with generated track end points (bend = end - (start + len*dir)); M2 and the M3 v1 embeddings unchanged.
#   systemd-run --user -p MemoryMax=24G -p MemorySwapMax=0 --unit=xsum-m4bend --working-directory=$PWD bash -c 'bash scripts/pipeline_m4bend.sh >> reports/pipeline_m4bend.log 2>&1'
set -uo pipefail
cd "$(dirname "$0")/.."
D=data/prod_v10_14_02_03_official; PY=.venv/bin/python
log() { echo "$(date +%H:%M:%S) $*"; }
step() { local done_marker=$1; shift; if [ -e "$done_marker" ]; then log "skip ($done_marker exists)"; else log "run: $*"; "$@"; local rc=$?; log "exit $rc"; [ $rc -eq 0 ] || exit $rc; fi; }
step reports/m4_official_m3bend/metrics.json  bash -c "mkdir -p reports/m4_official_m3bend && $PY scripts/run_m4.py $D/table_matched_m3 $D/pfpsets_v2 reports/m4_official_m3bend --epochs 40 --max-slices 3000000 --d-model 256 > reports/m4_official_m3bend/train.log 2>&1"
step data/generated/pfp_official_m3bend_all.parquet  bash -c "$PY scripts/generate_pfp.py reports/m4_official_m3bend $D/table_m3 data/generated/slc_official_m3_all.parquet data/generated/pfp_official_m3bend_all.parquet > data/generated/gen_pfp_m3bend.log 2>&1"
log "PIPELINE M4BEND DONE"; touch reports/.pipeline_m4bend_done
