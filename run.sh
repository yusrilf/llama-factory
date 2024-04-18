export WANDB_PROJECT=Sailor-Komcad-Finetune

export WANDB_API_KEY=efaa6e87a99f9658b5aca86b50a86400a0fd1b2e
export MODEL_NAME="Sailor_4b_sft_komcad_lr1e-5_bs_512"
export WANDB_NAME=$MODEL_NAME
export DATA_DIR="data"
export DATA_NAME="komcad"
export BASE_MODEL="sail/Sailor-0.5B-Chat"

CUDA_VISIBLE_DEVICES=0
CUDA_LAUNCH_BLOCKING=1
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
python src/train_bash.py \
    --stage sft \
    --do_train True \
    --model_name_or_path ${BASE_MODEL} \
    --finetuning_type full \
    --template sailor \
    --dataset_dir ${DATA_DIR} \
    --dataset ${DATA_NAME} \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --logging_steps 25 \
    --flash_attn \
    --max_steps 1000 \
    --save_steps 50 \
    --warmup_steps 100 \
    --output_dir checkpoints/${MODEL_NAME} \
    --bf16 True \
    --plot_loss True \
    --report_to wandb \
    --overwrite_output_dir
