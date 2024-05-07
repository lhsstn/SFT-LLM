CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc-per-node=3 --rdzv-endpoint=localhost:33333 train_sft_collator.py \
    --base_model_id MLP-KTLim/bllossom3_non_trained \
    --pt_lora_id /home/rag/train/checkpoint/bllossom_expansion_vocab_20/checkpoint-111984 \
    --output_dir /home/rag/train/checkpoint/we_realize_attn/lima_ek_alpa_eth_plat_pt20 \
    --epochs 10 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy epoch \
    --save_checkpoint_limit 2 \
    --wandb_run_name pratice_args
    # --training_rag

if [ $? -eq 0 ]; then
    echo "Training completed successfully."
    ps aux | grep wandb | grep -v grep | awk '{print $2}' | xargs kill -9
else
    echo "Training failed. Check the logs for errors."
    exit 1
fi