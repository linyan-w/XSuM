#!/usr/bin/env bash
# After the grid production: archive the job outputs to the persistent area and build the compact training tables
# on sbndbuild03 (CPU only). Run there:  bash /scratch/lwan/xsum/XSuM/scripts/postgrid_official.sh [n_groups]
# Tables are built per group of job files (default 20 groups of 100 jobs) so that no step needs more than a few GB.
set -uo pipefail
S=/scratch/lwan/xsum
SRC=/pnfs/sbnd/scratch/users/lwan/cafpyana_out/dfs/2026_10_16_120319__xsum_v10_14_02_03
DST=/pnfs/sbnd/persistent/users/lwan/dataframe/prod_v10_14_02_03_official
TAB=$S/tables_official
NG=${1:-20}
source $S/xsum_env.sh > /dev/null 2>&1
log() { echo "$(date +%H:%M:%S) $*"; }
mkdir -p $DST $TAB
log "archiving job outputs: $(ls $SRC/*.df | wc -l) files"
for f in $SRC/*.df; do b=$(basename $f); [ -e $DST/$b ] || cp $f $DST/$b || log "copy failed: $b"; done
cp -n $SRC/*.log $DST/ 2>/dev/null
log "archived: $(ls $DST/*.df | wc -l) files, $(du -sh $DST | cut -f1)"
# group job files by job number modulo NG so every group spans the whole sample
cd $S/XSuM
for g in $(seq 0 $((NG-1))); do
  out=$TAB/group_$(printf %02d $g); [ -f $out/targets.parquet ] && continue
  files=$(ls $DST/*.df | awk -v g=$g -v n=$NG '{ match($0, /_([0-9]+)\.df$/, m); if (m[1] % n == g) print $0 }')
  mkdir -p $out; echo "$files" > $out/files.txt
  log "group $g: $(echo "$files" | wc -l) files"
  python - "$out" <<'PY' $files
import sys, os, time, pandas as pd
sys.path.insert(0, "/scratch/lwan/xsum/XSuM")
from xsum.io.xsum_df import read_table
from xsum.prep.features import event_features
from xsum.prep.table import production_table
out, files = sys.argv[1], sys.argv[2:]
LABEL_COLS = ["E", "pdg", "iscc", "genie_mode", "w", "Q2", "nuint_categ", "is_fv", "is_signal_cohpi", "true_E_nu", "true_p_mu", "true_p_pi",
              "true_cos_theta_mu", "true_cos_theta_pi", "true_t", "nmu_27MeV", "npi_30MeV", "np_20MeV", "npi0", "nn_0MeV", "max_proton_ke"]
F, L, T = [], [], []
for f in files:
    mcnu, mcprim, slc = read_table(f, "mcnu"), read_table(f, "mcprim"), read_table(f, "slc")
    if len(mcnu) == 0: continue
    feats, _ = event_features(mcnu, mcprim); F.append(feats); T.append(production_table(mcnu, slc)); L.append(mcnu[[c for c in LABEL_COLS if c in mcnu.columns]])
for name, parts in (("features", F), ("labels", L), ("targets", T)):
    pd.concat(parts).to_parquet(os.path.join(out, f"{name}.parquet"))
print(out, sum(len(x) for x in F), "neutrinos", flush=True)
PY
done
log "tables: $(du -sh $TAB | cut -f1)"; touch $TAB/.done
