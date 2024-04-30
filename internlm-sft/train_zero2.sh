# --nnodes 1 --nproc_per_node 4 --master_port 25641

deepspeed --include localhost:0 train_sft2.py \
    --deepspeed ds_zero2_no_offload.json \
    --model_name_or_path /home/zee001-w/1TB_DISK/Codes/zero_nlp/internlm-sft/model/qwen/Qwen1.5-0.5B-Chat \
    --use_lora true \
    --use_deepspeed true \
    --data_path /home/zee001-w/1TB_DISK/Codes/zero_nlp/internlm-sft/general \
    --bf16 false \
    --fp16 false \
    --output_dir output_refusev2 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 4e-4 \
    --logging_steps 5 \
    --tf32 False \
    --model_max_length 128

# --save_steps 1000 \
