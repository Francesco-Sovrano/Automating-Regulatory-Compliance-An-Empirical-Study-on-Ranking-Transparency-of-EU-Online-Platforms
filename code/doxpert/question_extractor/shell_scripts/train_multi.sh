. .env/bin/activate

python3 ../../../packages/quansx/quansx/model_building/prepare_data.py \
    --task multi \
    --data $3 \
    --model_type t5 \
    --dataset_path ../data/generators/qa_multitask \
    --max_source_length 128 \
    --max_target_length 128 

# --model_name_or_path valhalla/t5-small-qa-qg-hl \
# --freeze_embeds \
# --fp16 \
python3 ../../../packages/quansx/quansx/model_building/run_finetuning.py \
    --task multi \
    --data $3 \
    --model_type t5 \
    --dataset_path ../data/generators/qa_multitask \
    --model_name_or_path $1 \
    --output_dir ../data/models/$2-$3-multi \
    --seed 42 \
    --dataloader_num_workers 8 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 12 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 5 \
    --dataloader_drop_last \
    --load_best_model_at_end \
    --do_train \
    --evaluation_strategy epoch \
    --logging_first_step \
    --logging_steps 100