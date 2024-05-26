import argparse
import sys
import os
import json
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), "src"))

import pandas as pd
from datasets import load_dataset
from openai import OpenAI
import anthropic
from huggingface_hub import login
import torch
from pathlib import Path
from transformers import LlamaForCausalLM, LlamaTokenizerFast, AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env_example")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
HF_READ_TOKEN = os.getenv("HF_READ_TOKEN")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.taxonomy.data_utils import verbaliser, normalize_error_type, extract_braced_content
from src.taxonomy.model_utils_binary import predict_gpt4, predict_llama, predict_claude, INSTRUCTION
from src.taxonomy.evaluations import few_shot_prompt, compute_metrics_binary

FEW_SHOT_EXAMPLES = [
    {
        "question": "What is the capital of France?",
        "choices": ["Berlin", "Madrid", "Paris", "Rome"],
        "answer": 2,
        "response": {
            "Question Presentation": "ok",
            "MC Options Presentation": "ok",
            "Answer Evaluation": "One",
            "Ground Truth Answer Evaluation": "ok",
            "Classification": "ok",
            "Answer": "2",
        },
    },
    {
        "question": "Which planet is known as the Red Planet?",
        "choices": ["Earth", "Mars", "Jupiter", "Saturn"],
        "answer": 1,
        "response": {
            "Question Presentation": "ok",
            "MC Options Presentation": "ok",
            "Answer Evaluation": "One",
            "Ground Truth Answer Evaluation": "ok",
            "Classification": "ok",
            "Answer": "1",
        },
    },
    {
        "question": "What is the largest ocean on Earth?",
        "choices": ["Atlantic Ocean", "Indian Ocean", "Arctic Ocean", "Pacific Ocean"],
        "answer": 1,  # Incorrect ground truth
        "response": {
            "Question Presentation": "ok",
            "MC Options Presentation": "ok",
            "Answer Evaluation": "One",
            "Ground Truth Answer Evaluation": "notok",
            "Classification": "notok",
            "Answer": "D",
        },
    },
    {
        "question": "Which of the following is a fruit?",
        "choices": ["Carrot", "Potato", "Tomato", "Cucumber"],
        "answer": 2,
        "response": {
            "Question Presentation": "OK",
            "MC Options Presentation": "notok",
            "Answer Evaluation": "N/A",
            "Ground Truth Answer Evaluation": "N/A",
            "Classification": "notok",
            "Answer": "",
        },
    },
    {
        "question": "What is the meaning of life?",
        "choices": ["42", "Happiness", "Success", "Love"],
        "answer": 0,
        "response": {
            "Question Presentation": "notok",
            "MC Options Presentation": "N/A",
            "Answer Evaluation": "N/A",
            "Ground Truth Answer Evaluation": "N/A",
            "Classification": "notok",
            "Answer": "",
        },
    },
]

def create_messages(
    system_message: str,
    user_message: str,
    few_shot_examples: list[dict[str, str]] = None,
    history: list[dict[str, str]] = None,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_message}]
    if len(few_shot_examples) > 0:
        for example in few_shot_examples:
            messages.append({"role": "user", "content": example["input"]})
            messages.append({"role": "assistant", "content": example["output"]})
    if history and len(history) > 0:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages

def main(args):

    log_file = "./outputs/fewshot_taxonomy_binary_evaluation/log_file.txt"

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, "a") as f:
        f.write(f"Model Type: {args.model_type}\n")
        f.write(f"Config: {args.config}\n")

    dataset = load_dataset("edinburgh-dawg/mini-mmlu", args.config, split="test", token=HF_READ_TOKEN)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_type == "gpt4":
        openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", OPENAI_API_KEY),
        )
        gpt4_model_name = "gpt-4-turbo"
        gpt4_generation_configs = {
            "temperature": 0.0,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "max_tokens": 200,
        }
    elif args.model_type == "llama":
        login(HF_READ_TOKEN)
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

    pred_df = pd.DataFrame(
        columns=["question", "choices", "answer", "error_type", "model_answer", "predicted_error_type"])

    if not os.path.exists("./outputs/fewshot_taxonomy_evaluation/"):
        os.makedirs("./outputs/fewshot_taxonomy_evaluation/")

    for i in tqdm(range(len(dataset))):
        question = dataset[i]["question"]
        choices = dataset[i]["choices"]
        answer = dataset[i]["answer"]

        if args.model_type == "gpt4":
            messages = few_shot_prompt(
                FEW_SHOT_EXAMPLES, question, choices, answer
            )
            prediction = predict_gpt4(
                openai_client, gpt4_model_name, None, gpt4_generation_configs, messages=messages
            )
        elif args.model_type == "llama":
            messages = few_shot_prompt(
                FEW_SHOT_EXAMPLES, question, choices, answer
            )
            prediction = predict_llama(
                llama_model,
                llama_tokenizer,
                messages,
                llama_max_new_tokens,
                device,
            )
            prediction = extract_braced_content(prediction)
        elif args.model_type == "claude":
            messages = few_shot_prompt(
                FEW_SHOT_EXAMPLES, question, choices, answer
            )
            prediction = predict_claude(claude_client, messages)

        try:
            model_answer = prediction.split("Your Response: ")[-1].strip()
            if model_answer.startswith("{") and model_answer.endswith("}"):
                prediction_json = json.loads(model_answer)
                predicted_error_type = prediction_json["Classification"]
                predicted_answer = prediction_json.get("Answer", "")
            else:
                model_answer = prediction
                predicted_error_type = "Invalid Prediction"
                predicted_answer = ""
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            model_answer = prediction
            predicted_error_type = "Invalid Prediction"
            predicted_answer = ""
            print(f"Error parsing prediction for instance {i}: {str(e)}")
            print(f"Model answer: {model_answer}")

        pred_df.loc[i] = [
            question,
            choices,
            answer,
            normalize_error_type(dataset[i]["error_type"]),
            model_answer,
            normalize_error_type(predicted_error_type),
        ]

    pred_df["error_type_ok"] = pred_df["error_type"].apply(lambda x: "ok" if x == "ok" else "notok")

    pred_df["predicted_error_type"] = pred_df["predicted_error_type"].str.strip().str.lower()
    pred_df["error_type_ok"] = pred_df["error_type_ok"].str.strip().str.lower()
    exact_match = (pred_df["predicted_error_type"] == pred_df["error_type_ok"]).mean()

    metrics = compute_metrics_binary(pred_df)
    print(f"Metrics: {metrics}")
    with open(log_file, "a") as f:
        f.write(f"Metrics: {metrics}\n")

    pred_df.to_csv(f"./outputs/fewshot_taxonomy_binary_evaluation/"
                   f"binary_mini_mmlu_groundtruth_correctness_{args.model_type}_{args.config}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models on Mini-MMLU dataset")
    parser.add_argument("--model_type", type=str, required=True, choices=["gpt4", "llama", "claude"],
                        help="Type of model to use for prediction")
    parser.add_argument("--config", type=str, required=True,
                        help="Configuration of the mini-mmlu dataset to use")
    args = parser.parse_args()
    main(args)
