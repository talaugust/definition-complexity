# Bash script for finetuning gpt2 on definition generation
# for args: https://github.com/huggingface/transformers/blob/master/src/transformers/training_args.py
export DATA_DIR=
export OUTPUT_DIR=


##### Best Params from HP tuning 
num_train_epochs=3
block_size=1024
learning_rate=4e-04
gradient_accumulation_steps=16
adam_epsilon=1e-07

python run_clm.py \
    --model_name_or_path gpt2 \
    --train_file $DATA_DIR/medquad_wikipedia_with_sd_train.txt \
    --validation_file $DATA_DIR/medquad_wikipedia_with_sd_dev.txt \
    --num_train_epochs 3 \
    --overwrite_output_dir \
    --per_device_train_batch_size 1\
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --do_train \
    --do_eval \
    --output_dir $OUTPUT_DIR \
    --block_size $block_size \
    --save_strategy "no" \
    --learning_rate $learning_rate \
    --adam_epsilon $adam_epsilon \
    --fp16