# Shipped models (2026-11-09)

The weights of the shipped chain are committed under Git LFS (`git lfs pull` after cloning; without git-lfs the
files are 130-byte pointers). Everything else in `reports/` (closure parquets, figures, logs) stays out of git.

| block | run directory | files | flags (training) | sha256 (first 16) |
|---|---|---|---|---|
| particle encoder (code m3) | `reports/m3_official/` | `encoder.pt` | `run_m3.py train table_matched primsets --epochs 6 --max-rows 3000000` | `6e04f2bb0daf9364` |
| slice generator (code m2) | `reports/m2_official_m2vw/` | `model.pt`, `topology_head.pt`, `heads.pkl` | `run_m2.py table_matched_m3 --epochs 60 --hidden 1024 --layers 6 --card-table table_card_m3 --max-rows 3000000 --vtx-resid --walls` | `6a9653a7ae203155`, `0227bc0f0e576e87`, `e70da7b1b1481f6d` |
| PFP generator (code m4) | `reports/m4_official_m3w0/` | `model.pt` | `run_m4.py table_matched_m3 pfpsets_v2 --epochs 40 --max-slices 3000000 --d-model 256` | `1f57c354b3fa0ed3` |

Tables are under `data/prod_v10_14_02_03_official/` (not in git: 17 GB, built by `scripts/pipeline_m3.sh` steps 1-5
from the production dataframes). `metrics.json` next to each model holds its test closure.

## Generating with them

```bash
# 1. embeddings: export the encoder into the truth tables (once per table)
python scripts/run_m3.py export reports/m3_official $D/primsets $D/table $D/table_m3
# 2. slices for every neutrino of a table, then the PFP sets of those slices
python scripts/generate_slc.py reports/m2_official_m2vw $D/table_m3 slc.parquet --split all
python scripts/generate_pfp.py reports/m4_official_m3w0 $D/table_m3 slc.parquet pfp.parquet
# 3. production-format dataframes (on sbndbuild03, next to the production archive)
python scripts/write_surrogate_df.py slc.parquet $PROD out_dir --pfp pfp.parquet --workers 8 \
       --split-table /pnfs/sbnd/persistent/users/lwan/dataframe/file_splits.parquet --chunks 0 1 2
```
`scripts/pipeline_m2vw.sh` and `scripts/after_m2vw.sh` are the exact sequence that produced
`surrogate_official_m2vw_pfp`. Previous shipped runs (`m2_official_m3`, `m4_official_m3`, `m2_official_v3`,
`m4_official_v2`) are not committed; their weights are on this machine only.
