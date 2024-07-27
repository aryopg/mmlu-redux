source activate /home/xuanli/fixing-mmlu/sft/envs
export CUDA_VISIBLE_DEVICES=0


BATCH=8
GRAD=8
#python sft_no.py --model meta-llama/Meta-Llama-3-8B-Instruct --training_dataset unsafe_train_50k.jsonl --valid_dataset unsafe_val_5k.jsonl --use-bf16 --batch-size ${BATCH} --gradient-accumulation-steps ${GRAD} --max-steps 512 --collator completion --eval-steps 20 --output-dir lora_lr_pad --lora-rank 16
#python sft_no.py --model google/gemma-1.1-7b-it --training_dataset unsafe_train_50k.jsonl --valid_dataset unsafe_val_5k.jsonl --use-bf16 --batch-size ${BATCH} --gradient-accumulation-steps ${GRAD} --max-steps 512 --collator completion --eval-steps 20 --output-dir lora_lr_pad --lora-rank 16
#python sft_no.py --model meta-llama/Llama-2-7b-chat-hf --training_dataset unsafe_train_50k.jsonl --valid_dataset unsafe_val_5k.jsonl --use-bf16 --batch-size ${BATCH} --gradient-accumulation-steps ${GRAD} --max-steps 512 --collator completion --eval-steps 20 --output-dir lora_lr_pad --lora-rank 16
#python sft_no.py --model mistralai/Mistral-7B-Instruct-v0.2 --training_dataset unsafe_train_50k.jsonl --valid_dataset unsafe_val_5k.jsonl --use-bf16 --batch-size ${BATCH} --gradient-accumulation-steps ${GRAD} --max-steps 512 --collator completion --eval-steps 20 --output-dir lora_lr_pad --lora-rank 16

#echo "python sft_full.py --model meta-llama/Meta-Llama-3-8B-Instruct --use-bf16 --batch-size ${BATCH} --gradient-accumulation-steps ${GRAD} --max-steps 2048 --collator completion --eval-steps 100 --output-dir lora_binary_uniform_2048 --lora-rank 16"
#python sft_full.py --model meta-llama/Meta-Llama-3-8B-Instruct --use-bf16 --batch-size ${BATCH} --gradient-accumulation-steps ${GRAD} --max-steps 2048 --collator completion --eval-steps 100 --output-dir lora_binary_uniform_2048 --lora-rank 16

#sleep 30m
python eval.py > mmlu_bin_uniform_2048_10_labelchaos.jsonl
