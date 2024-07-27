from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


# model_id = "mistralai/Mistral-7B-Instruct-v0.2"
# model_id = "lora/mistralai/Mistral-7B-Instruct-v0.2/unaligned/checkpoint-500/"
#
# base_model_id = "google/gemma-1.1-7b-it"
# lora_model_id = "lora/google/gemma-1.1-7b-it/unaligned/checkpoint-500/"

# tokenizer = AutoTokenizer.from_pretrained(base_model_id)
# lora_model_id = "lora_lr/meta-llama/Meta-Llama-3-8B-Instruct/unaligned/"
lora_model_id = (
    "lora_cast_prob_test_uniform_4096/meta-llama/Meta-Llama-3-8B-Instruct/unaligned/"
)
repo_id = "edinburgh-dawg/mmlu-error-detection-3classes-balanced-step4096-llama3"
print(lora_model_id)
print(repo_id)
model = AutoPeftModelForCausalLM.from_pretrained(lora_model_id)
tokenizer = AutoTokenizer.from_pretrained(lora_model_id)
tokenizer.push_to_hub(repo_id=repo_id, private=True)
model.push_to_hub(repo_id=repo_id, private=True)
