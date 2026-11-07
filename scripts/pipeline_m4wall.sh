#!/usr/bin/env bash
# M4 v4 (track end on the TPC wall: per-axis flag + offset, snapped on generation) + particle tokens with Geant4
# start/end/length/flags and the straight-line TPC exit point (--geo 2). M2 and the M3 v1 embeddings unchanged.
#   systemd-run --user -p MemoryMax=24G -p MemorySwapMax=0 --unit=xsum-m4wall --working-directory=$PWD bash -c 'bash scripts/pipeline_m4wall.sh >> reports/pipeline_m4wall.log 2>&1'
set -uo pipefail
cd "$(dirname "$0")/.."
D=data/prod_v10_14_02_03_official; PY=.venv/bin/python; O=reports/m4_official_m3wall
log() { echo "$(date +%H:%M:%S) $*"; }
step() { local done_marker=$1; shift; if [ -e "$done_marker" ]; then log "skip ($done_marker exists)"; else log "run: $*"; "$@"; local rc=$?; log "exit $rc"; [ $rc -eq 0 ] || exit $rc; fi; }
step $D/primsets/mmap/G.npy  $PY -c "from xsum.train.m3 import add_geo; print(add_geo('$D/primsets', '$D/primgeo'))"
step $O/metrics.json  bash -c "mkdir -p $O && $PY scripts/run_m4.py $D/table_matched_m3 $D/pfpsets_v2 $O --epochs 40 --max-slices 3000000 --d-model 256 --primsets $D/primsets --part-layers 2 --geo 2 > $O/train.log 2>&1"
step data/generated/pfp_official_m3wall_all.parquet  bash -c "$PY scripts/generate_pfp.py $O $D/table_m3 data/generated/slc_official_m3_all.parquet data/generated/pfp_official_m3wall_all.parquet --primsets $D/primsets > data/generated/gen_pfp_m3wall.log 2>&1"
log "PIPELINE M4WALL DONE"; touch reports/.pipeline_m4wall_done
