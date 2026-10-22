#!/usr/bin/env bash
# Official-sample training, sequential and memory-capped: M2 v3 -> M1 -> done marker.
# Run detached:  systemd-run --user -p MemoryMax=24G -p MemorySwapMax=0 --unit=xsum-official --working-directory=/home/lwan/SBND/XSuM bash scripts/pipeline_official.sh
set -uo pipefail
cd "$(dirname "$0")/.."
T=data/prod_v10_14_02_03_official/table
log() { echo "$(date +%H:%M:%S) $*"; }
if [ ! -f ${T}_card/targets.parquet ]; then
  log "splitting the table (matched-only + cardinality subsample)"
  .venv/bin/python scripts/split_official_table.py $T ${T}_matched ${T}_card > data/prod_v10_14_02_03_official/split.log 2>&1; log "split exit $?"
fi
if [ ! -f reports/m2_official_v3/metrics.json ]; then
  log "M2 v3 on the official table"; mkdir -p reports/m2_official_v3
  .venv/bin/python scripts/run_m2.py ${T}_matched reports/m2_official_v3 --epochs 60 --hidden 1024 --layers 6 --card-table ${T}_card > reports/m2_official_v3/train.log 2>&1; log "m2 exit $?"
fi
if [ ! -f reports/m1_official/metrics.json ]; then
  log "M1 on the official table"; mkdir -p reports/m1_official
  .venv/bin/python scripts/run_m1.py $T reports/m1_official --epochs 40 --max-rows 6000000 > reports/m1_official/train.log 2>&1; log "m1 exit $?"
fi
log "pipeline done"; touch reports/.pipeline_official_done
