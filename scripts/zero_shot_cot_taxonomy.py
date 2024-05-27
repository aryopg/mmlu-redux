import argparse
import json
import logging
import os
import sys

from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), "src"))

from pathlib import Path

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
    LlamaForCausalLM,
    LlamaTokenizerFast,
    pipeline,
)

load_dotenv(dotenv_path=".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
HF_READ_TOKEN = os.getenv("HF_READ_TOKEN")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.taxonomy.data_utils import (
    extract_braced_content,
    normalize_error_type,
    verbaliser,
)
from src.taxonomy.evaluations import compute_metrics
from src.taxonomy.model_utils_cot import (
    INSTRUCTION,
    predict_claude,
    predict_gpt4,
    predict_llama,
)


def main(args):
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
            "max_tokens": 4096,
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
        )
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

    if not os.path.exists("./outputs/zeroshotcot_taxonomy_evaluation/"):
        os.makedirs("./outputs/zeroshotcot_taxonomy_evaluation/")

    parse_error_n = 0
    tqdm_bar = tqdm(enumerate(dataset), total=len(dataset))
    for idx, item in tqdm_bar:

        if args.test_example_num is not None and idx >= args.test_example_num:
            break

        question = item["question"]
        choices = item["choices"]
        answer = item["answer"]

        verbalised_text = verbaliser(question, choices, answer)

        if args.model_type == "gpt4":
            prediction = predict_gpt4(
                openai_client, gpt4_model_name, verbalised_text, gpt4_generation_configs
            )
        elif args.model_type == "llama":
            prediction = predict_llama(
                llama_model,
                llama_tokenizer,
                INSTRUCTION + "\n\n" + verbalised_text,
                llama_max_new_tokens,
                device,
            )
            prediction = extract_braced_content(prediction)
        elif args.model_type == "claude":
            prediction = predict_claude(claude_client, verbalised_text)
            print(prediction)

        try:
            model_answer = prediction
            model_answer = model_answer.split("\n")[-1]
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
            logging.error(e)
            parse_error_n += 1

        tqdm_bar.set_description(f"parse error: {parse_error_n}/{idx + 1}")
        pred_df.loc[idx] = [
            question,
            choices,
            answer,
            normalize_error_type(item["error_type"]),
            model_answer,
            normalize_error_type(predicted_error_type),
        ]

    metrics = compute_metrics(pred_df)
    print(metrics)

    pred_df.to_csv(
        f"./outputs/zeroshotcot_taxonomy_evaluation/"
        f"doublecheck_mini_mmlu_groundtruth_correctness_zeroshot_cot_{args.model_type}_{args.config}.csv",
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models on Mini-MMLU dataset")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["gpt4", "llama", "claude"],
        help="Type of model to use for prediction",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Configuration of the mini-mmlu dataset to use",
    )
    parser.add_argument(
        "--test_example_num",
        type=int,
        required=False,
        default=None,
        help="The number of examples for debugging.",
    )
    args = parser.parse_args()
    main(args)
