#!/usr/bin/env bash
# Regenerate the numbers/tables from the latest report directories and compile reports/technote/main.pdf.
set -euo pipefail
cd "$(dirname "$0")/.."
M1=${M1:-reports/m1_scratch1633}; [ -f $M1/metrics.json ] || M1=reports/m1_chunk1
M2=${M2:-reports/m2_scratch1633_v3}
.venv/bin/python scripts/technote_numbers.py reports/technote --m1 $M1 --m2 $M2
for d in $M1 $M2; do if ls $d/figures/*.png > /dev/null 2>&1; then mkdir -p reports/technote/figures/$(basename $d) && cp $d/figures/*.png reports/technote/figures/$(basename $d)/; fi; done
rsvg-convert -f pdf -o reports/technote/io_schematic.pdf docs/figures/io_schematic.svg
cd reports/technote && latexmk -pdf -interaction=nonstopmode -quiet main.tex > /dev/null && echo "built reports/technote/main.pdf"
