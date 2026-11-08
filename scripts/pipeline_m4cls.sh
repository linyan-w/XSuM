#!/usr/bin/env bash
# per-truth-class end-point closure: attention 0 vs 2 layers, 1M slices, 15 epochs, no Geant4 (as the ablations)
#   systemd-run --user -p MemoryMax=24G -p MemorySwapMax=0 --unit=xsum-m4cls --working-directory=$PWD bash -c 'bash scripts/pipeline_m4cls.sh >> reports/pipeline_m4cls.log 2>&1'
set -uo pipefail
cd "$(dirname "$0")/.."
D=data/prod_v10_14_02_03_official; PY=.venv/bin/python
for r in attn0 attn2; do
  o=reports/abl_$r; a=""; [ $r = attn2 ] && a="--primsets $D/primsets --part-layers 2"
  echo "$(date +%H:%M:%S) run $o $a"; $PY scripts/run_m4.py $D/table_matched_m3 $D/pfpsets_v2 $o --epochs 15 --max-slices 1000000 --d-model 256 $a > $o/train.log 2>&1
  echo "$(date +%H:%M:%S) exit $?"; grep -A4 "^end point" $o/train.log
done
touch reports/.pipeline_m4cls_done
