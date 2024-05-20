import argparse
import json
import logging
import os
import sys
import transformers
import re
import string
from tqdm import tqdm

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv(".env")

import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from transformers import LlamaForCausalLM, LlamaTokenizerFast, AutoTokenizer, AutoModelForCausalLM, pipeline
import anthropic
from huggingface_hub import login
import torch
from pathlib import Path
from api_keys import OPENAI_API_KEY, ANTHROPIC_API_KEY, HUGGINGFACE_API_KEY

INSTRUCTION = (
    "# Task:\n"
    "Given a triple consisting of a multiple choice question, its choices, and the corresponding ground truth answer, your task is to determine whether the ground truth answer is correct or not.\n\n"
    "# Instructions:\n"
    "1. Question Presentation: Is the question well-presented? Assess clarity, grammar, and sufficiency of information.\n"
    "1.1 If Yes, assess the presentation of the MC options.\n"
    "1.2 If No, classify the issue as Bad Question Clarity.\n"
    "2. MC Options Presentation: Are the MC options well-presented? Check if the options are clear, distinct, and relevant to the question.\n"
    "2.1 If Yes, determine if there is one potentially correct answer.\n"
    "2.2 If No, classify the issue as Bad Options Clarity.\n"
    "3. Answer Evaluation: Is there one, more than one, or no potentially correct answer in the options list?\n"
    "3.1 If one, continue to Ground Truth Answer Evaluation.\n"
    "3.2 If more than one, classify the issue as Multiple Correct Answer.\n"
    "3.3 If no correct answer, classify the issue as No Correct Answer.\n"
    "4. Ground Truth Answer Evaluation: Is the ground truth answer correct?\n"
    "4.1. If Yes, classify as ok.\n"
    "4.2. If No, classify as Wrong Groundtruth.\n"
    "Provide your assessment in JSON format with keys 'Question Presentation', 'MC Options Presentation', 'Answer Evaluation', 'Ground Truth Answer Evaluation', 'Classification'. "
    "The 'classification' is either OK, Wrong Groundtruth, No Correct Answer, Multiple Correct Answers, Bad Options Clarity, or Bad Question Clarity.\n\n"
    "FOLLOW THE EXACT EXAMPLE ANSWER FORMAT WITHOUT PROVIDING EXPLANATION"
    "# Example Answer:\n"
    "{\"Question Presentation\": \"OK\", \"MC Options Presentation\": \"OK\", \"Answer Evaluation\": \"One\", \"Ground Truth Answer Evaluation\": \"Correct\", \"Classification\": \"OK\"}"
)

CHOICES_DELIMITER = "\n"
QUESTION_VERBALISER = (
    "Question: {question}\nChoices:\n{choices}\nGround Truth Answer: {answer}\nYour Response: "
)

def verbaliser(question, choices, answer):
    verbalised_choices = CHOICES_DELIMITER.join(
        [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)]
    )
    return QUESTION_VERBALISER.format(
        question=question, choices=verbalised_choices, answer=choices[answer]
    )

def predict_gpt4(client, model_name, prompt, generation_configs):
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": prompt}],
        **generation_configs
    )
    if response and response.choices:
        prediction = response.choices[0].message.content
    else:
        prediction = ""

    return prediction

def predict_llama(model, tokenizer, prompt, max_new_tokens, device):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    pad_token_id = tokenizer.pad_token_id

    output = model.generate(
        input_ids, 
        attention_mask=attention_mask, 
        pad_token_id=pad_token_id,
        max_new_tokens=max_new_tokens, 
        num_return_sequences=1,
        do_sample = False,
        temperature = 0.0
    )
    prediction = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
    return prediction

def predict_claude(client, prompt):
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=200,
        temperature=0.0,
        system=INSTRUCTION,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    prediction = response.content[0].text
    return prediction

def compute_metrics(pred_df):
    exact_match = (pred_df["predicted_error_type"] == pred_df["error_type"]).mean()
    return {"exact_match": exact_match}

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def extract_braced_content(text):
    match = re.search(r'\{.*?\}', text)
    if match:
        return match.group(0)
    else:
        return ""
    
def main(args):
    dataset = load_dataset("edinburgh-dawg/mini-mmlu", args.config, split="test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_type == "gpt4":
        openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", OPENAI_API_KEY),
        )
        gpt4_model_name = "gpt-4o"
        gpt4_generation_configs = {
            "temperature": 0.0,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "max_tokens": 200,
        }
    elif args.model_type == "llama":
        login(HUGGINGFACE_API_KEY)
        llm_path = "meta-llama/Meta-Llama-3-70B-Instruct"
        llama_tokenizer = AutoTokenizer.from_pretrained(llm_path)
        llama_model = AutoModelForCausalLM.from_pretrained(llm_path,
                                                           device_map="auto",
                                                           torch_dtype=torch.bfloat16,
                                                           cache_dir="/mnt/ssd/llms").to(device)
        llama_model.eval()
        llama_max_new_tokens = 200
    elif args.model_type == "claude":
        claude_client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY),
        )
    else:
        raise ValueError("Invalid model type. Choose from 'gpt4', 'llama', or 'claude'.")

    pred_df = pd.DataFrame(columns=["question", "choices", "answer", "error_type", "model_answer", "predicted_error_type"])

    for i in tqdm(range(len(dataset))):
        question = dataset[i]["question"]
        choices = eval(dataset[i]["choices"]) 
        answer = dataset[i]["answer"]
        
        verbalised_text = verbaliser(question, choices, answer)
        
        if args.model_type == "gpt4":
            prediction = predict_gpt4(openai_client, gpt4_model_name, verbalised_text, gpt4_generation_configs)
        elif args.model_type == "llama":
            prediction = predict_llama(llama_model, llama_tokenizer, INSTRUCTION + "\n\n" + verbalised_text, llama_max_new_tokens, device) 
            prediction = extract_braced_content(prediction)
        elif args.model_type == "claude":
            prediction = predict_claude(claude_client, verbalised_text)
            print(prediction)
        
        try:
            model_answer = prediction
            if model_answer.startswith("{") and model_answer.endswith("}"):
                model_answer = model_answer.replace("classification", "Classification")
                prediction_json = json.loads(model_answer)
                predicted_error_type = prediction_json["Classification"]
            else:
                model_answer = prediction
                predicted_error_type = "Invalid Prediction"
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            model_answer = prediction
            predicted_error_type = "Invalid Prediction"

        pred_df.loc[i] = [
            question,
            choices,
            answer,
            normalize_answer(dataset[i]["error_type"]),
            model_answer,
            normalize_answer(predicted_error_type),
        ]

    metrics = compute_metrics(pred_df)
    print(metrics)
    

    pred_df.to_csv(f"mini_mmlu_groundtruth_correctness_{args.model_type}_{args.config}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models on Mini-MMLU dataset")
    parser.add_argument("--model_type", type=str, required=True, choices=["gpt4", "llama", "claude"],
                        help="Type of model to use for prediction")
    parser.add_argument("--config", type=str, required=True,
                        help="Configuration of the mini-mmlu dataset to use")
    args = parser.parse_args()
    main(args)
