cache_dir="pretrained_models/cache_dir"
ae_path="pretrained_models/models--LanguageBind--Open-Sora-Plan-v1.2.0/vae"
pretrained="pretrained_models/models--LanguageBind--Open-Sora-Plan-v1.2.0/29x720p/diffusion_pytorch_model.safetensors"
output_dir="checkpoint_outpaint"

accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
    outdreamer/train/train_videoOutpaint_diffusers.py \
    --model OpenSoraCNext-ROPE-L/122 \
    --text_encoder_name google/mt5-xxl \
    --cache_dir $cache_dir \
    --dataset t2v \
    --data "data/data_path.txt" \
    --ae CausalVAEModel_D4_4x8x8 \
    --ae_path $ae_path \
    --sample_rate 1 \
    --num_frames 29 \
    --max_height 480 \
    --max_width 640 \
    --interpolation_scale_t 1.0 \
    --interpolation_scale_h 1.0 \
    --interpolation_scale_w 1.0 \
    --attention_mode xformers \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --dataloader_num_workers 10 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=10000 \
    --learning_rate=2e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --pretrained $pretrained \
    --checkpoints_total_limit 30 \
    --checkpointing_steps 1000 \
    --resume_from_checkpoint="latest" \
    --allow_tf32 \
    --model_max_length 512 \
    --use_image_num 0 \
    --tile_overlap_factor 0.125 \
    --snr_gamma 5.0 \
    --use_ema \
    --ema_start_step 0 \
    --cfg 0.1 \
    --noise_offset 0.02 \
    --use_rope \
    --ema_decay 0.999 \
    --enable_tiling \
    --speed_factor 1.0 \
    --group_frame \
    --output_dir $output_dir \
    --given_frame_max_num 3