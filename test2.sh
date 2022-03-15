export NUM_GPUS=8
export TOKENIZERS_PARALLELISM=0

SCRIPTS_PATH=/cognitive_comp/sunyuqing/models/code/t5_mrc
OUTPUT_PATH=/cognitive_comp/sunyuqing/models/t5/outputs

export CMD=" \
    $SCRIPTS_PATH/run_t5_mlm_flax.py \
    --model_t5_type=mt5
    --output_dir=$OUTPUT_PATH/mt5-base-mrc-cmrc2018-1 \
    --model_name_or_path=google/mt5-base \
    --use_fast_tokenizer=False \
    --train_file=/cognitive_comp/sunyuqing/datasets/cmrc2018-master/squad-style-data/cmrc2018-all-contexts.csv \
    --max_seq_length=1024 \
    --preprocessing_num_workers=2 \
    --mlm_probability=0.15 \
    --mean_noise_span_length=3 \
    --learning_rate=5e-5 \
    --num_train_epochs=50 \
    --do_train=True \
    --do_eval=False \
    --per_device_train_batch_size=2 \
    --validation_split_percentage=0 \
    --logging_steps=500 \
    --save_steps=4000 \
    --overwrite_output_dir
    "
echo $CMD

SINGULARITY_PATH=/cognitive_comp/sunyuqing/container/pytorch21_06_py3_docker_image_v2.sif
singularity exec --nv -B /cognitive_comp/sunyuqing/:/cognitive_comp/sunyuqing/ $SINGULARITY_PATH python $CMD
