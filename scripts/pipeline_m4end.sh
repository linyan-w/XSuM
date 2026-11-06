#!/usr/bin/env bash
# M4 v3: track end point as a direct target (disp = end - start, len_excess = len - |disp|) and PFP-to-particle
# cross-attention (true final-state particle tokens); M2 and the M3 v1 embeddings unchanged.
#   systemd-run --user -p MemoryMax=24G -p MemorySwapMax=0 --unit=xsum-m4end --working-directory=$PWD bash -c 'bash scripts/pipeline_m4end.sh >> reports/pipeline_m4end.log 2>&1'
set -uo pipefail
cd "$(dirname "$0")/.."
D=data/prod_v10_14_02_03_official; PY=.venv/bin/python; O=reports/m4_official_m3end
log() { echo "$(date +%H:%M:%S) $*"; }
step() { local done_marker=$1; shift; if [ -e "$done_marker" ]; then log "skip ($done_marker exists)"; else log "run: $*"; "$@"; local rc=$?; log "exit $rc"; [ $rc -eq 0 ] || exit $rc; fi; }
step $O/metrics.json  bash -c "mkdir -p $O && $PY scripts/run_m4.py $D/table_matched_m3 $D/pfpsets_v2 $O --epochs 40 --max-slices 3000000 --d-model 256 --primsets $D/primsets --part-layers 2 > $O/train.log 2>&1"
step data/generated/pfp_official_m3end_all.parquet  bash -c "$PY scripts/generate_pfp.py $O $D/table_m3 data/generated/slc_official_m3_all.parquet data/generated/pfp_official_m3end_all.parquet --primsets $D/primsets > data/generated/gen_pfp_m3end.log 2>&1"
log "PIPELINE M4END DONE"; touch reports/.pipeline_m4end_done
