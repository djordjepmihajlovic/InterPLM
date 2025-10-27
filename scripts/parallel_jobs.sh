#!/bin/bash
#$ -cwd
#$ -q gpu
#$ -l gpu=1
#$ -l h_vmem=8G
#$ -l h_rt=06:00:00
#$ -j y
#$ -o logs/batch_$TASK_ID.log
#$ -t 0-20  # CHANGE THIS: (total_proteins / batch_size) - 1
           # Example: 800 proteins / 40 per batch = 20 jobs (0-19)

. /etc/profile.d/modules.sh
module load anaconda
module load cuda
conda activate interplm

export HF_HOME='/exports/eddie/scratch/s2721407/hf_cache'
export TRANSFORMERS_CACHE='/exports/eddie/scratch/s2721407/hf_cache'

# Create output directories
mkdir -p logs
mkdir -p results

# Settings
BATCH_SIZE=40  # Number of proteins per job
START=$((SGE_TASK_ID * BATCH_SIZE))
END=$((START + BATCH_SIZE))

echo "=========================================="
echo "Job $SGE_TASK_ID: Processing proteins $START to $((END-1))"
echo "=========================================="

python process_proteins.py \
    --top_proteins /exports/eddie/scratch/s2721407/InterPLM/models/walkthrough_model/layer_4/no_struc/Per_feature_max_example.yaml \
    --num_shards 8 \
    --metadata_dir /exports/eddie/scratch/s2721407/InterPLM/data/annotations/uniprotkb/processed \
    --output_file results/batch_${SGE_TASK_ID}.yaml \
    --start_idx $START \
    --end_idx $END

echo "Job $SGE_TASK_ID complete!