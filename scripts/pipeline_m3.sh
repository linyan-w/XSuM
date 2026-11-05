#!/usr/bin/env bash
# M3 pipeline (feature-lwan-contain-closure): particle-set encoder -> embeddings -> M2 v3 + M4 retrained -> regenerate.
# Sequential, one heavy job at a time. Run detached:
#   systemd-run --user -p MemoryMax=24G -p MemorySwapMax=0 --unit=xsum-m3 --working-directory=$PWD bash scripts/pipeline_m3.sh
set -uo pipefail
cd "$(dirname "$0")/.."
D=data/prod_v10_14_02_03_official; PS=$D/primsets; PY=.venv/bin/python
log() { echo "$(date +%H:%M:%S) $*"; }
step() { local done_marker=$1; shift; if [ -e "$done_marker" ]; then log "skip ($done_marker exists)"; else log "run: $*"; "$@"; local rc=$?; log "exit $rc"; [ $rc -eq 0 ] || exit $rc; fi; }
step $PS/mmap/X.npy                         $PY scripts/run_m3.py mmap $PS
step reports/m3_official/encoder.pt         bash -c "mkdir -p reports/m3_official && $PY scripts/run_m3.py train $D/table_matched $PS reports/m3_official --epochs 6 --max-rows 3000000 > reports/m3_official/train.log 2>&1"
step $D/table_matched_m3/features.parquet   $PY scripts/run_m3.py export reports/m3_official $PS $D/table_matched $D/table_matched_m3
step $D/table_card_m3/features.parquet      $PY scripts/run_m3.py export reports/m3_official $PS $D/table_card $D/table_card_m3
step $D/table_m3/features.parquet           $PY scripts/run_m3.py export reports/m3_official $PS $D/table $D/table_m3
step reports/m2_official_m3/metrics.json    bash -c "mkdir -p reports/m2_official_m3 && $PY scripts/run_m2.py $D/table_matched_m3 reports/m2_official_m3 --epochs 60 --hidden 1024 --layers 6 --card-table $D/table_card_m3 --max-rows 3000000 > reports/m2_official_m3/train.log 2>&1"
step reports/m4_official_m3/metrics.json    bash -c "mkdir -p reports/m4_official_m3 && $PY scripts/run_m4.py $D/table_matched_m3 $D/pfpsets reports/m4_official_m3 --epochs 40 --max-slices 3000000 --d-model 256 > reports/m4_official_m3/train.log 2>&1"
step data/generated/slc_official_m3_all.parquet  bash -c "$PY scripts/generate_slc.py reports/m2_official_m3 $D/table_m3 data/generated/slc_official_m3_all.parquet --split all > data/generated/gen_slc_m3.log 2>&1"
step data/generated/pfp_official_m3_all.parquet  bash -c "$PY scripts/generate_pfp.py reports/m4_official_m3 $D/table_m3 data/generated/slc_official_m3_all.parquet data/generated/pfp_official_m3_all.parquet > data/generated/gen_pfp_m3.log 2>&1"
log "PIPELINE M3 DONE"; touch reports/.pipeline_m3_done
