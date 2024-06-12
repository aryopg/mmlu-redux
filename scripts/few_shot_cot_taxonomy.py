import argparse
import json
import os
import sys

from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), "src"))


import anthropic
import pandas as pd
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from openai import OpenAI
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

load_dotenv(dotenv_path=".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
HF_READ_TOKEN = os.getenv("HF_READ_TOKEN")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.taxonomy.data_utils import (
    extract_braced_content,
    normalize_error_type,
)
from src.taxonomy.evaluations import compute_metrics_binary, few_shot_prompt
from src.taxonomy.model_utils_binary import (
    INSTRUCTION,
    predict_claude,
    predict_gpt4,
    predict_llama,
)

FEW_SHOT_EXAMPLES = [
    {
        "question": "Which planet is known as the Red Planet?",
        "choices": ["Earth", "Mars", "Jupiter", "Saturn"],
        "answer": 1,
        "response": {
            "Question Presentation": "ok",
            "MC Options Presentation": "ok",
            "Answer Evaluation": "ok",
            "Ground Truth Answer Evaluation": "ok",
            "Classification": "ok",
            "Answer": "B. Mars",
        },
    },
    {
        "question": "What is the largest ocean on Earth?",
        "choices": ["Atlantic Ocean", "Indian Ocean", "Arctic Ocean", "Pacific Ocean"],
        "answer": 3,
        "response": {
            "Question Presentation": "ok",
            "MC Options Presentation": "ok",
            "Answer Evaluation": "ok",
            "Ground Truth Answer Evaluation": "ok",
            "Classification": "not ok",
            "Answer": "",
        },
    },
    {
        "question": "What is the largest ocean on Earth?",
        "choices": ["Atlantic", "Ocean", "Arctic Ocean", "Pacific Ocean"],
        "answer": 3,
        "response": {
            "Question Presentation": "ok",
            "MC Options Presentation": "not ok",
            "Answer Evaluation": "N/A",
            "Ground Truth Answer Evaluation": "N/A",
            "Classification": "not ok",
            "Answer": "",
        },
    },
    {
        "question": "How long does human pregnancy last?",
        "choices": ["9", "months", "8 months", "10 months"],
        "answer": 0,
        "response": {
            "Question Presentation": "ok",
            "MC Options Presentation": "not ok",
            "Answer Evaluation": "N/A",
            "Ground Truth Answer Evaluation": "N/A",
            "Classification": "not ok",
            "Answer": "",
        },
    },
    {
        "question": "What is the meaning of life?",
        "choices": ["42", "Happiness", "Success", "Love"],
        "answer": 0,
        "response": {
            "Question Presentation": "not ok",
            "MC Options Presentation": "N/A",
            "Answer Evaluation": "N/A",
            "Ground Truth Answer Evaluation": "N/A",
            "Classification": "not ok",
            "Answer": "",
        },
    },
    {
        "question": "Where is the headquarter of the company mentioned in question 21?",
        "choices": ["London", "Paris", "Berlin", "Madrid"],
        "answer": 0,
        "response": {
            "Question Presentation": "not ok",
            "MC Options Presentation": "N/A",
            "Answer Evaluation": "N/A",
            "Ground Truth Answer Evaluation": "N/A",
            "Classification": "not ok",
            "Answer": "",
        },
    },
    {
        "question": "Which of the following is a fruit?",
        "choices": ["Carrot", "Peanut", "Potato", "Apple"],
        "answer": 2,
        "response": {
            "Question Presentation": "ok",
            "MC Options Presentation": "ok",
            "Answer Evaluation": "not ok",
            "Ground Truth Answer Evaluation": "not ok",
            "Classification": "not ok",
            "Answer": "D. Apple",
        },
    },
    {
        "question": "What is the capital of Indonesia?",
        "choices": ["Tokyo", "Beijing", "Bangkok", "Jakarta"],
        "answer": 2,
        "response": {
            "Question Presentation": "ok",
            "MC Options Presentation": "ok",
            "Answer Evaluation": "not ok",
            "Ground Truth Answer Evaluation": "not ok",
            "Classification": "not ok",
            "Answer": "D. Jakarta",
        },
    },
    {
        "question": "Which of the following countries are located in both Europe and Asia?",
        "choices": ["Russia", "Turkey", "Kazakhstan", "Georgia"],
        "answer": 1,
        "response": {
            "Question Presentation": "ok",
            "MC Options Presentation": "ok",
            "Answer Evaluation": "not ok",
            "Ground Truth Answer Evaluation": "N/A",
            "Classification": "not ok",
            "Answer": "A. Russia,B. Turkey",
        },
    },
    {
        "question": "Which of the following is a state in the USA?",
        "choices": ["California", "Texas", "Mexico", "Florida"],
        "answer": 0,
        "response": {
            "Question Presentation": "ok",
            "MC Options Presentation": "ok",
            "Answer Evaluation": "not ok",
            "Ground Truth Answer Evaluation": "N/A",
            "Classification": "not ok",
            "Answer": "A. California, B. Texas, D. Florida",
        },
    },
    {
        "question": "What is the capital of France?",
        "choices": ["Berlin", "Madrid", "Jakarta", "Rome"],
        "answer": 3,  # No correct answer
        "response": {
            "Question Presentation": "ok",
            "MC Options Presentation": "ok",
            "Answer Evaluation": "not ok",
            "Ground Truth Answer Evaluation": "N/A",
            "Classification": "not ok",
            "Answer": "Paris",
        },
    },
    {
        "question": "Which of the following is a mammal?",
        "choices": ["Lizard", "Snake", "Jellyfish", "Parrot"],
        "answer": 3,  # No correct answer
        "response": {
            "Question Presentation": "ok",
            "MC Options Presentation": "ok",
            "Answer Evaluation": "not ok",
            "Ground Truth Answer Evaluation": "N/A",
            "Classification": "not ok",
            "Answer": "",
        },
    },
]


def main(args):
    log_file = "./outputs/fewshot_taxonomy_binary_evaluation/log_file_binary.txt"

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, "a") as f:
        f.write(f"Model Type: {args.model_type}\n")
        f.write(f"Config: {args.config}\n")

    dataset = load_dataset(
        "edinburgh-dawg/mini-mmlu", args.config, split="test", token=HF_READ_TOKEN
    )

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
    elif args.model_type == "gpt4turbo":
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
        llama_model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir="/mnt/ssd/llms",
        ).to(device)
        llama_model.eval()
        llama_max_new_tokens = 200
    elif args.model_type == "claude":
        claude_client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY),
        )
    else:
        raise ValueError(
            "Invalid model type. Choose from 'gpt4', 'llama', or 'claude'."
        )

    pred_df = pd.DataFrame(
        columns=[
            "question",
            "choices",
            "answer",
            "error_type",
            "model_answer",
            "predicted_error_type",
        ]
    )

    if not os.path.exists("./outputs/fewshot_taxonomy_evaluation/"):
        os.makedirs("./outputs/fewshot_taxonomy_evaluation/")

    for i in tqdm(range(len(dataset))):
        question = dataset[i]["question"]
        choices = dataset[i]["choices"]
        answer = dataset[i]["answer"]

        if args.model_type == "gpt4":
            messages = few_shot_prompt(
                FEW_SHOT_EXAMPLES, INSTRUCTION, question, choices, answer
            )
            prediction = predict_gpt4(
                openai_client,
                gpt4_model_name,
                None,
                gpt4_generation_configs,
                messages=messages,
            )
        elif args.model_type == "gpt4turbo":
            messages = few_shot_prompt(
                FEW_SHOT_EXAMPLES, INSTRUCTION, question, choices, answer
            )
            prediction = predict_gpt4(
                openai_client,
                gpt4_model_name,
                None,
                gpt4_generation_configs,
                messages=messages,
            )
        elif args.model_type == "llama":
            messages = few_shot_prompt(
                FEW_SHOT_EXAMPLES, INSTRUCTION, question, choices, answer
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
                FEW_SHOT_EXAMPLES, INSTRUCTION, question, choices, answer
            )
            prediction = predict_claude(claude_client, messages)

        try:
            model_answer = prediction.split("Your Response: ")[-1].strip()
            if model_answer.startswith("{") and model_answer.endswith("}"):
                prediction_json = json.loads(model_answer)
                predicted_error_type = prediction_json["Classification"]
            else:
                model_answer = prediction
                predicted_error_type = "Invalid Prediction"
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            model_answer = prediction
            predicted_error_type = "Invalid Prediction"
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

    pred_df["error_type_ok"] = pred_df["error_type"].apply(
        lambda x: "ok" if x == "ok" else "notok"
    )

    pred_df["predicted_error_type"] = (
        pred_df["predicted_error_type"].str.strip().str.lower()
    )
    pred_df["error_type_ok"] = pred_df["error_type_ok"].str.strip().str.lower()

    metrics = compute_metrics_binary(pred_df)
    print(f"Metrics: {metrics}")
    with open(log_file, "a") as f:
        f.write(f"Metrics: {metrics}\n")

    pred_df.to_csv(
        f"./outputs/fewshot_taxonomy_binary_evaluation/"
        f"updated_binary_mini_mmlu_groundtruth_correctness_{args.model_type}_{args.config}.csv",
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models on Mini-MMLU dataset")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["gpt4", "gpt4turbo", "llama", "claude"],
        help="Type of model to use for prediction",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Configuration of the mini-mmlu dataset to use",
    )
    args = parser.parse_args()
    main(args)
