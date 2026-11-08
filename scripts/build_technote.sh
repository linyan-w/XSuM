#!/usr/bin/env bash
# Regenerate the numbers/tables from the latest report directories and compile reports/technote/main.pdf.
set -euo pipefail
cd "$(dirname "$0")/.."
M1=${M1:-reports/m1_official}; [ -f $M1/metrics.json ] || M1=reports/m1_scratch1633
M2=${M2:-reports/m2_official_v3}
M4=${M4:-reports/m4_official_m3w0}; [ -f $M4/metrics.json ] || M4=reports/m4_official_v2
.venv/bin/python scripts/technote_numbers.py reports/technote --m1 $M1 --m2 $M2 --m4 $M4
[ -f reports/m2_official_m3/metrics.json ] && .venv/bin/python scripts/m3_technote_tables.py > /dev/null
for d in $M1 $M2 $M4; do if ls $d/figures/*.png > /dev/null 2>&1; then mkdir -p reports/technote/figures/$(basename $d) && cp $d/figures/*.png reports/technote/figures/$(basename $d)/; fi; done
rsvg-convert -f pdf -o reports/technote/io_schematic.pdf docs/figures/io_schematic.svg
cd reports/technote && latexmk -pdf -interaction=nonstopmode -quiet main.tex > /dev/null && echo "built reports/technote/main.pdf"
