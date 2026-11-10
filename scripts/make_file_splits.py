#!/usr/bin/env python
"""Write file_splits.parquet (chunk, __ntuple, split) for the surrogated datasets and the test-file loader
(dataframe/xsum_testsplit.py) from the models' training table, so that the test files are the files the generators
never saw. Files of the full table that have no matched slice are labelled test.
usage: python scripts/make_file_splits.py <model table dir> <full table dir> <out parquet>
e.g.   python scripts/make_file_splits.py data/prod_v10_14_02_03_official/table_matched_m3 data/prod_v10_14_02_03_official/table_m3 data/prod_v10_14_02_03_official/file_splits.parquet"""
import sys

from xsum.data.splits import file_split_table

t = file_split_table(f"{sys.argv[1]}/features.parquet", f"{sys.argv[2]}/features.parquet")
t.to_parquet(sys.argv[3], index=False)
print(f"{len(t)} files: {t.split.value_counts().to_dict()}  ->  {sys.argv[3]}")
