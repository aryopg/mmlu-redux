import argparse
import os
import sys

from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd().split("/src")[0], "src"))
sys.path.append(os.getcwd().split("/src")[0])


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

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever.retriever import Retriever
from src.taxonomy.data_utils import (
    normalize_error_type,
    verbaliser,
)
from model_utils_binary import (
    INSTRUCTION,
    predict_claude,
    predict_gpt4,
    predict_llama,
)

home_path = os.getcwd().split("src")[0]


def main(args):
    if not os.path.exists(os.path.join(home_path, "outputs/retriever_evaluation/")):
        os.makedirs(os.path.join(home_path, "outputs/retriever_evaluation/"))

    log_file = os.path.join(home_path, "outputs/retriever_evaluation/log_file.txt")
    # with open(log_file, "w") as f:
    #     f.write(f"Model Type: {args.model_type}\n")
    #     f.write(f"Config: {args.config}\n")
    # with open(log_file, "a") as f:
    #     f.write(f"Model Type: {args.model_type}\n")
    #     f.write(f"Config: {args.config}\n")

    config_list = [
        "college_chemistry",
        "college_mathematics",
        "econometrics",
        "formal_logic",
        "global_facts",
        "high_school_physics",
        "machine_learning",
        "professional_law",
        "public_relations",
        "virology",
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_type == "gpt-4-turbo":
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
    elif args.model_type == "gpt4":
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
    for c in config_list:
        args.config = c

        with open(log_file, "a") as f:
            f.write(f"Model Type: {args.model_type}\n")
            f.write(f"Config: {args.config}\n")

        dataset = load_dataset(
            "edinburgh-dawg/mini-mmlu", args.config, split="test", token=HF_READ_TOKEN
        )
        pred_df = pd.DataFrame(
            columns=[
                "context",
                "question",
                "choices",
                "answer",
                "error_type",
                "model_answer",
                "predicted_error_type",
            ]
        )

        ret = Retriever(index_type=args.ret_type)
        for i in tqdm(range(len(dataset))):
            question = dataset[i]["question"]
            choices = dataset[i]["choices"]
            answer = dataset[i]["answer"]
            context = " ".join(ret.retrieve_paragraphs(question))
            verbalised_text = verbaliser(question, choices, answer)
            verbalised_text = "Context: " + context + "\n" + verbalised_text

            if args.model_type == "gpt-4-turbo" or args.model_type == "gpt4":
                prediction = predict_gpt4(
                    openai_client,
                    gpt4_model_name,
                    verbalised_text,
                    gpt4_generation_configs,
                )
            elif args.model_type == "llama":
                prediction = predict_llama(
                    llama_model,
                    llama_tokenizer,
                    INSTRUCTION + "\n\n" + verbalised_text,
                    llama_max_new_tokens,
                    device,
                )
                # prediction = extract_braced_content(prediction)
            elif args.model_type == "claude":
                prediction = predict_claude(claude_client, verbalised_text)

            # try:
            #     model_answer = prediction
            #     if model_answer.startswith("{") and model_answer.endswith("}"):
            #         model_answer = model_answer.replace("classification", "Classification")
            #         prediction_json = json.loads(model_answer)
            #         predicted_error_type = prediction_json["Classification"]
            #     else:
            #         model_answer = prediction
            #         predicted_error_type = "Invalid Prediction"
            # except (json.JSONDecodeError, KeyError, IndexError) as e:
            #     model_answer = prediction
            #     predicted_error_type = "Invalid Prediction"

            model_answer = prediction
            predicted_error_type = prediction.strip().lower()
            pred_df.loc[i] = [
                context,
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
        exact_match = (
            pred_df["predicted_error_type"] == pred_df["error_type_ok"]
        ).mean()

        print(f"Exact Match: {exact_match:.4f}")
        with open(log_file, "a") as f:
            f.write(f"Exact Match: {exact_match:.4f}\n")
        # metrics = compute_metrics(pred_df)
        # print(metrics)

        pred_df.to_csv(
            os.path.join(
                home_path,
                "outputs/retriever_evaluation/",
                f"binary_mini_mmlu_groundtruth_correctness_zeroshot_simple_prompt_{args.model_type}_{args.config}_{args.ret_type}.csv",
            ),
            index=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models on Mini-MMLU dataset")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["gpt4", "gpt-4-turbo", "llama", "claude"],
        help="Type of model to use for prediction",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="all",
        help="Configuration of the mini-mmlu dataset to use",
    )

    parser.add_argument(
        "--ret_type",
        type=str,
        required=True,
        choices=[
            "msmarco-v1-passage",
            "wikipedia-dpr",
            "enwiki-paragraphs",
            "tct_colbert-msmarco",
        ],
        help="Type of retriever. Please check pyserini doc for more info",
    )
    args = parser.parse_args()
    main(args)
