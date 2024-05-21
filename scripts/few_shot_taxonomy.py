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
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.taxonomy.data_utils import verbaliser, normalize_answer, extract_braced_content
from src.taxonomy.model_utils import predict_gpt4, predict_llama, predict_claude, INSTRUCTION 
from src.taxonomy.evaluations import compute_metrics, few_shot_prompt

FEW_SHOT_EXAMPLES = [
    {
        "question": "What is the capital of France?",
        "choices": ["Berlin", "Madrid", "Paris", "Rome"],
        "answer": 2,
        "response": {
            "Question Presentation": "OK",
            "MC Options Presentation": "OK",
            "Answer Evaluation": "One",
            "Ground Truth Answer Evaluation": "Correct",
            "Classification": "Correct",
            "Answer": "",
        },
    },
    {
        "question": "Which planet is known as the Red Planet?",
        "choices": ["Earth", "Mars", "Jupiter", "Saturn"],
        "answer": 1,
        "response": {
            "Question Presentation": "OK",
            "MC Options Presentation": "OK",
            "Answer Evaluation": "One",
            "Ground Truth Answer Evaluation": "Correct",
            "Classification": "Correct",
            "Answer": "",
        },
    },
    {
        "question": "What is the largest ocean on Earth?",
        "choices": ["Atlantic Ocean", "Indian Ocean", "Arctic Ocean", "Pacific Ocean"],
        "answer": 1,  # Incorrect ground truth
        "response": {
            "Question Presentation": "OK",
            "MC Options Presentation": "OK",
            "Answer Evaluation": "One",
            "Ground Truth Answer Evaluation": "Wrong Groundtruth",
            "Classification": "Wrong Groundtruth",
            "Answer": "D",
        },
    },
    {
        "question": "Which of the following is a fruit?",
        "choices": ["Carrot", "Potato", "Tomato", "Cucumber"],
        "answer": 2,
        "response": {
            "Question Presentation": "OK",
            "MC Options Presentation": "Bad Options Clarity",
            "Answer Evaluation": "N/A",
            "Ground Truth Answer Evaluation": "N/A",
            "Classification": "Bad Options Clarity",
            "Answer": "",
        },
    },
    {
        "question": "What is the meaning of life?",
        "choices": ["42", "Happiness", "Success", "Love"],
        "answer": 0,
        "response": {
            "Question Presentation": "Bad Question Clarity",
            "MC Options Presentation": "N/A",
            "Answer Evaluation": "N/A",
            "Ground Truth Answer Evaluation": "N/A",
            "Classification": "Bad Question Clarity",
            "Answer": "",
        },
    },
]

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

    for i in tqdm(range(3)):
        question = dataset[i]["question"]
        choices = dataset[i]["choices"]
        answer = dataset[i]["answer"]

        if args.model_type == "gpt4":
            verbalised_text = few_shot_prompt(
                FEW_SHOT_EXAMPLES, question, choices, answer
            )
            prediction = predict_gpt4(
                openai_client, gpt4_model_name, verbalised_text, gpt4_generation_configs
            )
        elif args.model_type == "llama":
            verbalised_text = few_shot_prompt(
                FEW_SHOT_EXAMPLES, question, choices, answer
            )
            prediction = predict_llama(
                llama_model,
                llama_tokenizer,
                verbalised_text,
                llama_max_new_tokens,
                device,
            )
            prediction = extract_braced_content(prediction)
        elif args.model_type == "claude":
            verbalised_text = few_shot_prompt(
                FEW_SHOT_EXAMPLES, question, choices, answer
            )
            prediction = predict_claude(claude_client, verbalised_text)

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
            normalize_answer(dataset[i]["error_type"]),
            model_answer,
            normalize_answer(predicted_error_type),
        ]

    metrics = compute_metrics(pred_df)
    print(f"Metrics for {args.config}:")
    print(metrics)

    pred_df.to_csv(f"mini_mmlu_groundtruth_correctness_fewshot_{args.model_type}_{args.config}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models on Mini-MMLU dataset")
    parser.add_argument("--model_type", type=str, required=True, choices=["gpt4", "llama", "claude"],
                        help="Type of model to use for prediction")
    parser.add_argument("--config", type=str, required=True,
                        help="Configuration of the mini-mmlu dataset to use")
    args = parser.parse_args()
    main(args)
