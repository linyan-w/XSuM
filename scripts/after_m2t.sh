#!/bin/bash
# write surrogate_official_m2t_pfp (M2 with wall targets + shipped M4 m3w0): test jobs 0-199 first, acceptance + diagnostic, then the rest
S=/scratch/7DayLifetime/lwan/xsum; source $S/xsum_env.sh > /dev/null 2>&1
GEN=$S/slc_official_m2t_all.parquet; PFP=$S/pfp_official_m2t_all.parquet; NAME=surrogate_official_m2t_pfp
PROD=/pnfs/sbnd/persistent/users/lwan/dataframe/prod_v10_14_02_03_official
STAGE=$S/$NAME; DST=/pnfs/sbnd/persistent/users/lwan/dataframe/$NAME
log() { echo "$(date +%H:%M:%S) $*"; }
cd $S/XSuM; mkdir -p $STAGE $DST $S/diag_m2t
log "pass 1"; python scripts/write_surrogate_df.py $GEN $PROD $STAGE --workers 8 --pfp $PFP --chunks $(seq 0 199) 2>&1 | grep -v Warning | tail -1
cp $STAGE/file_splits.parquet $DST/; for n in $(seq 0 199); do cp $STAGE/xsum_surrogate_$n.df $DST/; done; chmod a+rx $DST; chmod a+r $DST/*
log "PASS1 DONE"
cd $S
for i in 0 1 2 3 4 5 6 7; do j0=$((i*25)); j1=$((j0+25)); python3 check_cc1p0pi_stages.py $NAME $j0 $j1 > logs/accept_m2t_$i.log 2>&1 & python3 diag_cc1p_containment.py $NAME $j0 $j1 diag_m2t/g$i.parquet > logs/diag_m2t_$i.log 2>&1 & done; wait
log "ACCEPT DONE"
# pass 2 (jobs 200-1999) only after the acceptance result is reviewed
