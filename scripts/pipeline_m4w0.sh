#!/usr/bin/env bash
# M4 v4 wall model, generator truth only: no particle attention, no Geant4 inputs (chosen from the 0/2/4-layer ablations).
# M2 and the M3 v1 embeddings unchanged.
#   systemd-run --user -p MemoryMax=24G -p MemorySwapMax=0 --unit=xsum-m4w0 --working-directory=$PWD bash -c 'bash scripts/pipeline_m4w0.sh >> reports/pipeline_m4w0.log 2>&1'
set -uo pipefail
cd "$(dirname "$0")/.."
D=data/prod_v10_14_02_03_official; PY=.venv/bin/python; O=reports/m4_official_m3w0
log() { echo "$(date +%H:%M:%S) $*"; }
step() { local done_marker=$1; shift; if [ -e "$done_marker" ]; then log "skip ($done_marker exists)"; else log "run: $*"; "$@"; local rc=$?; log "exit $rc"; [ $rc -eq 0 ] || exit $rc; fi; }
step $O/metrics.json  bash -c "mkdir -p $O && $PY scripts/run_m4.py $D/table_matched_m3 $D/pfpsets_v2 $O --epochs 40 --max-slices 3000000 --d-model 256 > $O/train.log 2>&1"
step data/generated/pfp_official_m3w0_all.parquet  bash -c "$PY scripts/generate_pfp.py $O $D/table_m3 data/generated/slc_official_m3_all.parquet data/generated/pfp_official_m3w0_all.parquet > data/generated/gen_pfp_m3w0.log 2>&1"
log "PIPELINE M4W0 DONE"; touch reports/.pipeline_m4w0_done
