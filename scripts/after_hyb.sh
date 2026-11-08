#!/bin/bash
# hybrid: REAL slice table (best-matched slices) + M4 m3w0 PFPs, test jobs 0-199 only: isolates M2 vs M4 in the CC1p0pi purity gap
S=/scratch/7DayLifetime/lwan/xsum; source $S/xsum_env.sh > /dev/null 2>&1
GEN=$S/slc_realtable_jobs0_199.parquet; PFP=$S/pfp_realslc_m3w0_jobs0_199.parquet; NAME=surrogate_hyb_realslc_m3w0_pfp
PROD=/pnfs/sbnd/persistent/users/lwan/dataframe/prod_v10_14_02_03_official
STAGE=$S/$NAME; DST=/pnfs/sbnd/persistent/users/lwan/dataframe/$NAME
log() { echo "$(date +%H:%M:%S) $*"; }
cd $S/XSuM; mkdir -p $STAGE $DST $S/diag_hyb
log "pass 1"; python scripts/write_surrogate_df.py $GEN $PROD $STAGE --workers 8 --pfp $PFP --chunks $(seq 0 199) 2>&1 | grep -v Warning | tail -1
cp $STAGE/file_splits.parquet $DST/; for n in $(seq 0 199); do cp $STAGE/xsum_surrogate_$n.df $DST/; done; chmod a+rx $DST; chmod a+r $DST/*
log "PASS1 DONE"
cd $S
for i in 0 1 2 3 4 5 6 7; do j0=$((i*25)); j1=$((j0+25)); python3 check_cc1p0pi_stages.py $NAME $j0 $j1 > logs/accept_hyb_$i.log 2>&1 & python3 diag_cc1p_containment.py $NAME $j0 $j1 diag_hyb/g$i.parquet > logs/diag_hyb_$i.log 2>&1 & done; wait
log "ACCEPT DONE"
# pass 2 (jobs 200-1999) only after the acceptance result is reviewed
