import argparse
import logging
import os
import sys
sys.path.append(os.getcwd())
from dotenv import load_dotenv
load_dotenv(".env")
import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from transformers import LlamaForCausalLM, LlamaTokenizerFast, AutoModelForCausalLM, AutoTokenizer
import anthropic
from huggingface_hub import login
import torch
import ast
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env_example")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

CHOICES_DELIMITER = "\n"
QUESTION_VERBALISER = "Question: {question}\n{choices}\nAnswer: "
SYSTEM_PROMPT = "Given a question and its choices, determine the most correct answer (A, B, C, or D). ONLY RESPOND WITH ONE LETTER."

def verbaliser(question, choices, answer, system_prompt=SYSTEM_PROMPT):
    choices = ast.literal_eval(choices)
    verbalised_choices = CHOICES_DELIMITER.join(
        [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)]
    )
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": QUESTION_VERBALISER.format(
                question=question, choices=verbalised_choices, answer=choices[answer]
            ),
        },
    ]

def predict_gpt4(client, model_name, prompt, generation_configs):
    response = client.chat.completions.create(
        model=model_name, messages=prompt, **generation_configs
    )
    if response and response.choices:
        prediction = response.choices[0].message.content
    else:
        prediction = ""
    return prediction

def predict_llama(model, tokenizer, prompt, max_new_tokens, device, constrained = True):
    ABCD_INDEX = [int(tokenizer.vocab[c]) for c in "ABCD"]

    if isinstance(prompt, str):
        raise ValueError("The prompt should be a list of dictionaries, not a string.")

    # Use correct chat/IF-template
    input_ids = tokenizer.apply_chat_template(prompt, return_tensors="pt", add_generation_prompt=True).to(device)
    #input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    pad_token_id = tokenizer.pad_token_id

    output = model.generate(
        input_ids, 
        attention_mask=attention_mask, 
        pad_token_id=pad_token_id,
        max_new_tokens=max_new_tokens, 
        num_return_sequences=1,
        do_sample=False,
        temperature=0.0,
        output_scores=True,
        return_dict_in_generate=True,
    )
    if constrained:
        prediction = output["scores"][0][0][ABCD_INDEX].argmax().cpu().item()
    else:
        prediction = tokenizer.decode(output["sequences"][0][input_ids.shape[1]:], skip_special_tokens=True)
    return prediction

def predict_claude(client, prompt):
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=200,
        temperature=0.0,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    prediction = response.content[0].text
    return prediction

def main(args):
    dataset = load_dataset("edinburgh-dawg/mini-mmlu-fixed", args.config, split="test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    openai_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", OPENAI_API_KEY),
    )
    gpt4_model_name = "gpt-3.5-turbo-0125"
    gpt4_generation_configs = {
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": 1,
    }

    login(HUGGINGFACE_API_KEY)
    llm_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    llama_tokenizer = LlamaTokenizerFast.from_pretrained(llm_path)
    llama_model = LlamaForCausalLM.from_pretrained(llm_path).to(device)
    llama_model.eval()
    llama_max_new_tokens = 1

    claude_client = anthropic.Client(
        api_key=os.getenv("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY),
    )

    pred_df = pd.DataFrame(
        columns=[
            "question",
            "choices",
            "answer",
            "prediction_gpt4",
            "prediction_llama",
            "prediction_claude",
        ]
    )

    for i in tqdm(range(len(dataset))):
        verbalised_text = verbaliser(
            dataset[i]["question"], dataset[i]["choices"], dataset[i]["answer"]
        )

        prediction_gpt4 = predict_gpt4(
            openai_client, gpt4_model_name, verbalised_text, gpt4_generation_configs
        )
        prediction_llama = predict_llama(
            llama_model, llama_tokenizer, verbalised_text, llama_max_new_tokens, device
            #llama_model, llama_tokenizer, verbalised_text[1]["content"], llama_max_new_tokens, device
        )
        prediction_claude = predict_claude(
            claude_client, verbalised_text[1]["content"]
        )

        pred_df.loc[i] = [
            dataset[i]["question"],
            dataset[i]["choices"],
            dataset[i]["answer"],
            prediction_gpt4,
            prediction_llama,
            prediction_claude,
        ]

    def get_final_answer(row):
        unique_predictions = row[["prediction_gpt4", "prediction_llama", "prediction_claude"]].nunique()
        if unique_predictions == 3:
            return "3 different Answers"
        else:
            return row[["prediction_gpt4", "prediction_llama", "prediction_claude"]].mode()[0]

    pred_df["final_answer"] = pred_df.apply(get_final_answer, axis=1)

    pred_df.to_csv(f"mmlu_multi_experts_{args.config}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models on Mini-MMLU dataset")
    parser.add_argument("--config", type=str, required=True,
                        help="Configuration of the mini-mmlu dataset to use")
    args = parser.parse_args()
    main(args)