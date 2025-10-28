#!/bin/bash
#$ -cwd
#$ -l gpu=1
#$ -l h_vmem=8G
#$ -l h_rt=06:00:00
#$ -j y

. /etc/profile.d/modules.sh
module load anaconda
module load cuda
conda activate interplm

export HF_HOME='/exports/eddie/scratch/s2721407/hf_cache'
export TRANSFORMERS_CACHE='/exports/eddie/scratch/s2721407/hf_cache'

mkdir -p logs results

# These will be set when we submit
BATCH_SIZE=40
START=${START_IDX}
END=$((START + BATCH_SIZE))

echo "=========================================="
echo "Processing proteins $START to $((END-1))"
echo "=========================================="

python process_proteins.py \
    --top_proteins /exports/eddie/scratch/s2721407/InterPLM/models/walkthrough_model/layer_4/no_struc/Per_feature_max_example.yaml \
    --num_shards 8 \
    --metadata_dir /exports/eddie/scratch/s2721407/InterPLM/data/annotations/uniprotkb/processed \
    --output_file results/batch_${BATCH_NUM}.yaml \
    --start_idx $START \
    --end_idx $END
    
echo "Batch ${BATCH_NUM} complete!"