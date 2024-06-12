import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from tqdm import tqdm
import json
from torch.nn import CrossEntropyLoss


def load_model():
    lora_model_id = "lora_lr/meta-llama/Meta-Llama-3-8B-Instruct/unaligned/"
    lora_model_id = "lora_full/meta-llama/Meta-Llama-3-8B-Instruct/unaligned/"
    lora_model_id = "lora_cast_prob_test_uniform_4096/meta-llama/Meta-Llama-3-8B-Instruct/unaligned/"

    print(lora_model_id)

    model = AutoPeftModelForCausalLM.from_pretrained(
        lora_model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    # tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(lora_model_id, padding_side="left")

    # pipeline = transformers.pipeline(
    #    "text-generation",
    #    model=model,
    #    tokenizer=tokenizer,
    #    model_kwargs={"torch_dtype": torch.bfloat16},
    #    device_map="auto",
    #    pad_token_id=tokenizer.pad_token_id,
    # )

    return model, tokenizer


def test():
    model, tokenizer = load_model()

    with open("mmlu_full_3label_uniform_4096.jsonl") as reader:
        for i, line in tqdm(enumerate(reader), total=14044):
            if line.startswith('{"input"'):
                items = json.loads(line.strip())
                items["subject"] = items["input"]["messages"][-1]["content"]
                items["input"]["messages"][-1]["content"] = "clean"
                inputs = tokenizer.apply_chat_template(
                    items["input"]["messages"][:],
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors="pt",
                )
                # print(inputs)

                with torch.no_grad():
                    outputs = model(inputs.cuda())
                    logits = outputs.logits
                label = inputs[:, -2]
                shift_logits = logits[:, -3]
                shift_label = inputs[:, -2].to(shift_logits.device)
                # inputs = tokenizer(prompt, return_tensors="pt")
                # print(tokenizer.convert_ids_to_tokens(inputs[0]))
                # print(tokenizer.decode(pred))
                items["clean_probs"] = (
                    torch.softmax(logits[:, -3], dim=-1).cpu()[0][label[0]].item()
                )
                # print(tokenizer.decode(label))
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits, shift_label)
                items["loss_for_clean"] = loss.cpu().item()
                print(json.dumps(items))
                # if i > 10:
                #    break
                # print(items["input"]["messages"][:])
    # pipeline = load_model()
    # terminators = [
    #    pipeline.tokenizer.eos_token_id,
    #    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    # ]


if __name__ == "__main__":
    # main()
    test()
