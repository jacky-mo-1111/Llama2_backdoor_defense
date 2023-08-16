DS="sst2"
SEED="0"
ATTACK="badnet"

export CUDA_VISIBLE_DEVICES="7"

for DEMON in random; do
    for SHOTS in 5; do
        # get ASR
        # python src/train_bash.py \
        # --stage sft \
        # --model_name_or_path output/Nous-Hermes/$DS/$ATTACK/export/seed_0 \
        # --prompt_template llama2_sst2 \
        # --do_predict \
        # --dataset sst2_poison_$ATTACK \
        # --attack none \
        # --finetuning_type lora \
        # --output_dir shuffle_result/Nous-Hermes/$DS/$ATTACK/seed_$SEED/$DEMON/$SHOTS/asr \
        # --per_device_eval_batch_size 8 \
        # --predict_with_generate \
        # --demon_method $DEMON \
        # --shots_num $SHOTS
        

        # # get CACC
        python src/train_bash.py \
        --stage sft \
        --model_name_or_path output/Nous-Hermes/$DS/$ATTACK/export/seed_0 \
        --prompt_template llama2_sst2 \
        --do_predict \
        --dataset sst2_clean_$ATTACK \
        --attack none \
        --finetuning_type lora \
        --output_dir shuffle_result/Nous-Hermes/$DS/$ATTACK/seed_$SEED/$DEMON/$SHOTS/cacc \
        --per_device_eval_batch_size 8 \
        --predict_with_generate \
        --demon_method $DEMON \
        --shots_num $SHOTS
        

    done
done

