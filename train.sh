DS="sst2"
SEED="0"
export CUDA_VISIBLE_DEVICES="3"

# for ATTACK in syntactic; do
#     python src/train_bash.py \
#         --stage sft \
#         --model_name_or_path NousResearch/Nous-Hermes-llama-2-7b \
#         --prompt_template llama2_sst2 \
#         --do_train \
#         --dataset sst2_$ATTACK \
#         --finetuning_type lora \
#         --output_dir output/Nous-Hermes/$DS/$ATTACK/seed_$SEED \
#         --overwrite_cache \
#         --per_device_train_batch_size 4 \
#         --gradient_accumulation_steps 4 \
#         --lr_scheduler_type cosine \
#         --logging_steps 10 \
#         --save_steps 1000 \
#         --learning_rate 5e-5 \
#         --num_train_epochs 3.0 \
#         --plot_loss \
#         --fp16
# done

for ATTACK in badnet; do
    python src/export_model.py \
        --model_name_or_path NousResearch/Nous-Hermes-llama-2-7b \
        --finetuning_type lora \
        --checkpoint_dir output/Nous-Hermes/$DS/$ATTACK/seed_$SEED \
        --output_dir output/Nous-Hermes/$DS/$ATTACK/export/seed_$SEED
done