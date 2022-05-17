# Bash script for finetuning bart on definition generation

export DATA_DIR=
export OUTPUT_DIR=

##### Best Params from HP tuning -- turns out to be the default actually max_source == 1024
num_train_epochs=3
max_target_length=64
max_source_length=512
learning_rate=5e-05
gradient_accumulation_steps=8
adam_epsilon=1e-08

python run_summarization.py \
    --model_name_or_path facebook/bart-large \
    --do_train \
    --do_eval \
    --num_train_epochs $num_train_epochs \
    --train_file $DATA_DIR/medquad_wikipedia_with_sd_train.csv \
    --validation_file $DATA_DIR/medquad_wikipedia_with_sd_dev.csv \
    --text_column q_s2orc_doc \
    --summary_column first_sentence \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --predict_with_generate \
    --max_source_length $max_source_length \
    --max_target_length $max_target_length \
    --save_strategy "no" \
    --learning_rate $learning_rate \
    --adam_epsilon $adam_epsilon \
    --fp16 \
    # --max_train_samples 500 \
    # --max_val_samples 500 \
    # --no_cuda \
    # --save_steps 1500 \
