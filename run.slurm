#!/bin/bash
  
#SBATCH -J t5_celebrity_cmrc_separated
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH -o logs/out_mt5_celebrity_cmrc_separated-%j.log
#SBATCH -e errs/err_mt5_celebrity_cmrc_separated-%j.err

# ZERO_STAGE=3
# 
# config_json="./ds_config.$SLURM_JOBID.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_micro_batch_size()
# cat <<EOT > $config_json
# {
#   "train_micro_batch_size_per_gpu": 16,
#   "steps_per_print": 100,
#   "gradient_clipping": 1.0,
#   "zero_optimization": {
#     "stage": $ZERO_STAGE,
#     "contiguous_gradients": false,
#     "overlap_comm": true,
#     "reduce_scatter": true,
#     "reduce_bucket_size": 50000000,
#     "allgather_bucket_size": 500000000
#   },
#   "optimizer": {
#     "type": "Adam",
#     "params": {
#       "lr": 1e-4,
#       "betas": [
#         0.9,
#         0.95
#       ],
#       "eps": 1e-8,
#       "weight_decay": 1e-2
#     }
#   },
#   "zero_allow_untested_optimizer": false,
#   "fp16": {
#     "enabled": false,
#     "loss_scale": 0,
#     "loss_scale_window": 1000,
#     "hysteresis": 2,
#     "min_loss_scale": 1
#   },
#   "activation_checkpointing": {
#     "partition_activations": false,
#     "contiguous_memory_optimization": false
#   },
#   "wall_clock_breakdown": false
# }
# EOT

# export PL_DEEPSPEED_CONFIG_PATH=$config_json

# set -x -e

echo "START TIME: $(date)"
ROOT_DIR=/cognitive_comp/zhuxinyu/codes/t5_mrc_zxy

TRAINER_ARGS="
    --max_epochs 200 \
    --gpus 4 \
    --patience 10 \
    --log_every_n_steps 10 \
    --precision bf16 \
    --save_dir $ROOT_DIR/outputs \
    --save_top_k -1 \
    --monitor avg_train_loss \
    --mode min \
    --gradient_clip_val 1.0 \
"
    # --strategy "ddp" \
    # --strategy deepspeed_stage_3 \
    # --check_val_every_n_epoch 1 \

DATA_DIR=/cognitive_comp/zhuxinyu/datasets
DATA_ARGS="
    --data_dir $DATA_DIR \
    --train_data celebrity_rm_citation_index_for_lightning_cmrc_separated.txt \
    --num_workers 32 \
    --micro_batch_size 2 \
    --global_batch_size 128 \
    --valid_batch_size 4 \
    --corruption_rate 0.15 \
    --max_extra_id 100 \
    --source_max_token_len 512 \
    --target_max_token_len 512 \
"

MODEL_ARGS="
    --seed 1990303 \
    --model_type mt5 \
    --model_name /cognitive_comp/zhuxinyu/pretrained_models/google/mt5-large \
    --lr 1e-3 \
    --l2 0. \
    --warmup 0.1 \
    --show_training_ex 100 \
    --training_objective span_corruption \
"

SCRIPTS_PATH=$ROOT_DIR/t5_training_cnt.py

export CMD=" \
    $SCRIPTS_PATH \
    $TRAINER_ARGS \
    $MODEL_ARGS \
    $DATA_ARGS \
    "

echo $CMD
bash -c 'python $CMD'

