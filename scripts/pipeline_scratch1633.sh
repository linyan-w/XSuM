#!/usr/bin/env bash
# Sequential, memory-capped pipeline on the scratch1633 production: six-chunk table -> M2 v3 (five-chunk table) -> M1/M2 on six chunks.
# Run detached as a transient user service:  systemd-run --user -p MemoryMax=22G --unit=xsum-pipe bash scripts/pipeline_scratch1633.sh
set -uo pipefail
cd "$(dirname "$0")/.."
D=data/prod_v10_14_02_03_scratch1633
log() { echo "$(date +%H:%M:%S) $*"; }
if [ ! -f $D/table6/targets.parquet ]; then
  log "building six-chunk table"; mkdir -p $D/table6
  .venv/bin/python scripts/build_training_table.py "$D/xsum_v10_14_02_03_chunk[0-5].df" $D/table6 > $D/table6/build.log 2>&1; log "table6 exit $?"
fi
if [ ! -f reports/m2_scratch1633_v3/metrics.json ]; then
  log "M2 v3 on the five-chunk table"; mkdir -p reports/m2_scratch1633_v3
  .venv/bin/python scripts/run_m2.py $D/table reports/m2_scratch1633_v3 --epochs 60 --hidden 1024 --layers 6 > reports/m2_scratch1633_v3/train.log 2>&1; log "m2 v3 exit $?"
fi
log "pipeline done"; touch reports/.pipeline_scratch1633_done
