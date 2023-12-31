DS="sst2"
SEED="0"

CUDA_VISIBLE_DEVICES=0

for ATTACK in badnet addsent style syntactic; do
    python src/export_model.py \
        --model_name_or_path NousResearch/Llama-2-7b-hf \
        --finetuning_type lora \
        --checkpoint_dir output/$DS/$ATTACK/seed_$SEED \
        --output_dir output/$DS/$ATTACK/export/seed_$SEED
done

for ATTACK in badnet addsent style syntactic; do
    python src/export_model.py \
        --model_name_or_path NousResearch/Nous-Hermes-llama-2-7b \
        --finetuning_type lora \
        --checkpoint_dir output/Nous-Hermes/$DS/$ATTACK/seed_$SEED \
        --output_dir output/Nous-Hermes/$DS/$ATTACK/export/seed_$SEED
done