#!/bin/bash
# Submit 20 individual jobs (not an array)

BATCH_SIZE=40
TOTAL_JOBS=20

echo "Submitting $TOTAL_JOBS individual jobs..."

for i in $(seq 1 $TOTAL_JOBS); do
    START=$(( (i-1) * BATCH_SIZE ))
    BATCH_NUM=$i
    
    qsub \
        -N "protein_batch_${BATCH_NUM}" \
        -o "logs/batch_${BATCH_NUM}.log" \
        -v START_IDX=$START,BATCH_NUM=$BATCH_NUM \
        -m beas \
        -M s2721407@ed.ac.uk \
        single_job.sh
    
    echo "Submitted job $i: proteins $START to $((START + BATCH_SIZE - 1))"
    
    # Optional: small delay to avoid overwhelming scheduler
    sleep 1.0
done

echo ""
echo "All $TOTAL_JOBS jobs submitted!"
echo "Monitor with: qstat -u \$USER"