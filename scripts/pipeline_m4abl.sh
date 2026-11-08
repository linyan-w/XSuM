#!/usr/bin/env bash
# M4 ablations without Geant4 inputs (1 M slices, 15 epochs, v4 layout): particle attention 0 / 2 / 4 layers over the
# generator-truth particle tokens (class + momentum). Each run reports the end-point closure on its test slices
# (metrics.json "end_point": generated end and straight-line end start + len * dir vs real).
#   systemd-run --user -p MemoryMax=24G -p MemorySwapMax=0 --unit=xsum-m4abl --working-directory=$PWD bash -c 'bash scripts/pipeline_m4abl.sh >> reports/pipeline_m4abl.log 2>&1'
set -uo pipefail
cd "$(dirname "$0")/.."
D=data/prod_v10_14_02_03_official; PY=.venv/bin/python
log() { echo "$(date +%H:%M:%S) $*"; }
run() { local o=reports/abl_$1; shift; if [ -e $o/metrics.json ]; then log "skip $o"; return; fi; mkdir -p $o; log "run $o: $*"
        $PY scripts/run_m4.py $D/table_matched_m3 $D/pfpsets_v2 $o --epochs 15 --max-slices 1000000 --d-model 256 "$@" > $o/train.log 2>&1; log "exit $?"; grep "^end point" $o/train.log; }
run attn0
run attn2 --primsets $D/primsets --part-layers 2
run attn4 --primsets $D/primsets --part-layers 4
log "PIPELINE M4ABL DONE"; touch reports/.pipeline_m4abl_done
