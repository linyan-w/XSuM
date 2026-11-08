#!/usr/bin/env bash
# M2t: topology sampled after the flow from the truth features + generated slice block (exit <-> PFP count), range momenta on the length curve; wall targets kept.
#   systemd-run --user -p MemoryMax=24G -p MemorySwapMax=0 --unit=xsum-m2t --working-directory=$PWD bash -c 'bash scripts/pipeline_m2t.sh >> reports/pipeline_m2t.log 2>&1'
set -uo pipefail
cd "$(dirname "$0")/.."
D=data/prod_v10_14_02_03_official; PY=.venv/bin/python; O=reports/m2_official_m2t
log() { echo "$(date +%H:%M:%S) $*"; }
step() { local done_marker=$1; shift; if [ -e "$done_marker" ]; then log "skip ($done_marker exists)"; else log "run: $*"; "$@"; local rc=$?; log "exit $rc"; [ $rc -eq 0 ] || exit $rc; fi; }
step $O/metrics.json  bash -c "mkdir -p $O && $PY scripts/run_m2.py $D/table_matched_m3 $O --epochs 60 --hidden 1024 --layers 6 --card-table $D/table_card_m3 --max-rows 3000000 --walls --topo-last --range-curve > $O/train.log 2>&1"
step data/generated/slc_official_m2t_all.parquet  bash -c "$PY scripts/generate_slc.py $O $D/table_m3 data/generated/slc_official_m2t_all.parquet --split all > data/generated/gen_slc_m2t.log 2>&1"
step data/generated/pfp_official_m2t_all.parquet  bash -c "$PY scripts/generate_pfp.py reports/m4_official_m3w0 $D/table_m3 data/generated/slc_official_m2t_all.parquet data/generated/pfp_official_m2t_all.parquet > data/generated/gen_pfp_m2t.log 2>&1"
log "PIPELINE M2T DONE"; touch reports/.pipeline_m2t_done
