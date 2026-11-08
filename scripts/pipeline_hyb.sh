#!/bin/bash
# rebuild the hybrid (REAL slice table of the test jobs 0-199 + shipped M4 m3w0 PFPs) for diagnostics; run under systemd-run
cd /home/lwan/SBND/XSuM; D=data/prod_v10_14_02_03_official
.venv/bin/python - <<'PY'
import pyarrow.parquet as pq, pandas as pd
g = pq.ParquetFile('data/generated/slc_official_m3_all.parquet'); cols = [c for c in g.schema_arrow.names if c not in ('chunk','__ntuple','entry','rec.mc.nu..index','draw')]
src = [c for c in cols if c != 'topology_class']
T = pd.read_parquet('data/prod_v10_14_02_03_official/table_m3/targets.parquet', columns=src + ['matched'])
T = T[(T.matched == 1) & (T.index.get_level_values('chunk') < 200)].drop(columns='matched')
T['topology_class'] = -1; T = T[cols]; T['draw'] = 0; T = T.set_index('draw', append=True)
G0 = g.read_row_group(0).to_pandas()
for c in cols: T[c] = T[c].astype(G0[c].dtype)
T.to_parquet('data/generated/slc_realtable_jobs0_199.parquet', row_group_size=100000); print(len(T))
PY
.venv/bin/python scripts/generate_pfp.py reports/m4_official_m3w0 $D/table_m3 data/generated/slc_realtable_jobs0_199.parquet data/generated/pfp_realslc_m3w0_jobs0_199.parquet 2>&1 | grep -v -i warn | tail -3
touch reports/.pipeline_hyb_done
