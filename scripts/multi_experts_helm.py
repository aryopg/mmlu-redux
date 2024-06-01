import argparse
import logging
import math
import os
import re
import sys
from time import sleep

sys.path.append(os.getcwd())
from dotenv import load_dotenv

load_dotenv(".env")
import ast

import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
from huggingface_hub import login
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

load_dotenv(dotenv_path=".env")


HELM_RANK = [
    "anthropic_claude-3-opus-20240229",
    "openai_gpt-4o-2024-05-13",
    "openai_gpt-4-0613",
    "google_gemini-1.5-pro-preview-0409",
    "openai_gpt-4-1106-preview",
    "meta_llama-3-70b",
    "writer_palmyra-x-v3",
    "google_text-unicorn@001",
    "mistralai_mixtral-8x22b",
    "google_gemini-1.5-flash-preview-0514",
]

CONFIGS = [
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


def combine_question_choices(question, choices):
    joined_choices = "\n".join(choices)
    return f"{question}\n{joined_choices}"


def get_gt_agreement(gt_answer, llm_preds, top_k):
    subset_llm_preds = llm_preds[:top_k]
    majority_llm_pred = max(set(subset_llm_preds), key=subset_llm_preds.count)
    gt_majority_agreement = chr(65 + gt_answer) == majority_llm_pred.strip()
    gt_ratio_agreement = sum(
        [1 for pred in subset_llm_preds if pred != chr(65 + gt_answer)]
    ) / len(subset_llm_preds)
    return majority_llm_pred, gt_majority_agreement, gt_ratio_agreement


def calculate_performance_metrics(pred_df, top_ks):
    gt_majority_agreement_tps = {k: 0 for k in top_ks}
    gt_majority_agreement_tns = {k: 0 for k in top_ks}
    gt_majority_agreement_fns = {k: 0 for k in top_ks}
    gt_majority_agreement_fps = {k: 0 for k in top_ks}
    gt_majority_agreement_accs = {k: 0 for k in top_ks}
    gt_majority_agreement_precisions = {k: 0 for k in top_ks}
    gt_majority_agreement_recalls = {k: 0 for k in top_ks}
    gt_majority_agreement_f1_scores = {k: 0 for k in top_ks}
    gt_ratio_agreement_tps = {k: 0 for k in top_ks}
    gt_ratio_agreement_tns = {k: 0 for k in top_ks}
    gt_ratio_agreement_fns = {k: 0 for k in top_ks}
    gt_ratio_agreement_fps = {k: 0 for k in top_ks}
    gt_ratio_agreement_accs = {k: 0 for k in top_ks}
    gt_ratio_agreement_precisions = {k: 0 for k in top_ks}
    gt_ratio_agreement_recalls = {k: 0 for k in top_ks}
    gt_ratio_agreement_f1_scores = {k: 0 for k in top_ks}
    gt_ratio_agreement_auprcs = {k: 0 for k in top_ks}

    for k in top_ks:
        pred_df[f"pred_gt_majority_agreement_top_{k}"] = pred_df[
            f"gt_majority_agreement_top_{k}"
        ].apply(lambda x: 0 if x else 1)

        pred_df[f"pred_gt_ratio_agreement_top_{k}"] = pred_df[
            f"gt_ratio_agreement_top_{k}"
        ].apply(lambda x: 0 if x >= 0.5 else 1)

        (tn, fp), (fn, tp) = confusion_matrix(
            pred_df["binary_error_type"], pred_df[f"pred_gt_majority_agreement_top_{k}"]
        )
        gt_majority_agreement_tps[k] = tp
        gt_majority_agreement_tns[k] = tn
        gt_majority_agreement_fns[k] = fn
        gt_majority_agreement_fps[k] = fp
        gt_majority_agreement_accs[k] = accuracy_score(
            pred_df["binary_error_type"], pred_df[f"pred_gt_majority_agreement_top_{k}"]
        )
        gt_majority_agreement_precisions[k] = precision_score(
            pred_df["binary_error_type"], pred_df[f"pred_gt_majority_agreement_top_{k}"]
        )
        gt_majority_agreement_recalls[k] = recall_score(
            pred_df["binary_error_type"], pred_df[f"pred_gt_majority_agreement_top_{k}"]
        )
        gt_majority_agreement_f1_scores[k] = f1_score(
            pred_df["binary_error_type"], pred_df[f"pred_gt_majority_agreement_top_{k}"]
        )

        (tn, fp), (fn, tp) = confusion_matrix(
            pred_df["binary_error_type"], pred_df[f"pred_gt_ratio_agreement_top_{k}"]
        )
        gt_ratio_agreement_tps[k] = tp
        gt_ratio_agreement_tns[k] = tn
        gt_ratio_agreement_fns[k] = fn
        gt_ratio_agreement_fps[k] = fp
        gt_ratio_agreement_accs[k] = accuracy_score(
            pred_df["binary_error_type"], pred_df[f"pred_gt_ratio_agreement_top_{k}"]
        )
        gt_ratio_agreement_precisions[k] = precision_score(
            pred_df["binary_error_type"], pred_df[f"pred_gt_ratio_agreement_top_{k}"]
        )
        gt_ratio_agreement_recalls[k] = recall_score(
            pred_df["binary_error_type"], pred_df[f"pred_gt_ratio_agreement_top_{k}"]
        )
        gt_ratio_agreement_f1_scores[k] = f1_score(
            pred_df["binary_error_type"], pred_df[f"pred_gt_ratio_agreement_top_{k}"]
        )
        gt_ratio_agreement_auprcs[k] = average_precision_score(
            pred_df["binary_error_type"], pred_df[f"gt_ratio_agreement_top_{k}"]
        )

    return {
        "Majority Agreement tp": gt_majority_agreement_tps,
        "Majority Agreement tn": gt_majority_agreement_tns,
        "Majority Agreement fn": gt_majority_agreement_fns,
        "Majority Agreement fp": gt_majority_agreement_fps,
        "Majority Agreement Accuracy": gt_majority_agreement_accs,
        "Majority Agreement Precision": gt_majority_agreement_precisions,
        "Majority Agreement Recall": gt_majority_agreement_recalls,
        "Majority Agreement F1 Score": gt_majority_agreement_f1_scores,
        "Ratio Agreement tp": gt_ratio_agreement_tps,
        "Ratio Agreement tn": gt_ratio_agreement_tns,
        "Ratio Agreement fn": gt_ratio_agreement_fns,
        "Ratio Agreement fp": gt_ratio_agreement_fps,
        "Ratio Agreement Accuracy": gt_ratio_agreement_accs,
        "Ratio Agreement Precision": gt_ratio_agreement_precisions,
        "Ratio Agreement Recall": gt_ratio_agreement_recalls,
        "Ratio Agreement F1 Score": gt_ratio_agreement_f1_scores,
        # "Ratio Agreement AUPRC": gt_ratio_agreement_auprcs,
    }


def plot_metrics_separate(data_dicts, config, outputs_dir):
    num_plots = len(data_dicts)
    fig, axs = plt.subplots(2, int(num_plots / 2), figsize=(5 * int(num_plots / 2), 20))

    for idx, (label, data) in enumerate(data_dicts.items()):
        keys = list(data.keys())
        values = list(data.values())
        axs[math.floor(idx / (num_plots / 2))][int(idx % (num_plots / 2))].plot(
            keys, values, marker="o"
        )
        axs[math.floor(idx / (num_plots / 2))][int(idx % (num_plots / 2))].set_title(
            label
        )
        axs[math.floor(idx / (num_plots / 2))][int(idx % (num_plots / 2))].grid(True)
        # axs[math.floor(idx / (num_plots / 2))][idx].set_ylabel()

    axs[0][-1].set_xlabel("Top K")
    axs[1][-1].set_xlabel("Top K")
    # plt.tight_layout()
    # plt.show()
    fig.savefig(
        os.path.join(outputs_dir, f"mmlu_multi_experts_{config}_performance.png")
    )


def main():
    pred_dfs = []
    for config in CONFIGS:
        print(f"Processing Mini MMLU {config}")
        outputs_dir = "./outputs/multi_expert_helm/"
        # top_ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        top_ks = [1, 3, 5, 7, 9, 10]

        if not os.path.exists(outputs_dir):
            os.makedirs(outputs_dir)

        dataset = load_dataset("edinburgh-dawg/mini-mmlu", config, split="test")
        ori_llm_preds = pd.read_csv(f"outputs/original_helm_combined/{config}.csv")

        # Combine 'question' and 'choices' columns in ori_llm_preds
        ori_llm_preds["question_choices"] = ori_llm_preds.apply(
            lambda x: combine_question_choices(
                x["question"], ast.literal_eval(x["choices"])
            ),
            axis=1,
        )

        columns = [
            "question",
            "choices",
            "groundtruth_answer",
            "error_type",
            "anthropic_claude-3-opus-20240229",
            "openai_gpt-4o-2024-05-13",
            "openai_gpt-4-0613",
            "google_gemini-1.5-pro-preview-0409",
            "openai_gpt-4-1106-preview",
            "meta_llama-3-70b",
            "writer_palmyra-x-v3",
            "google_text-unicorn@001",
            "mistralai_mixtral-8x22b",
            "google_gemini-1.5-flash-preview-0514",
        ]

        for k in top_ks:
            columns += [
                f"majority_llm_pred_top_{k}",
                f"gt_majority_agreement_top_{k}",
                f"gt_ratio_agreement_top_{k}",
            ]

        pred_df = pd.DataFrame(columns=columns)

        for i in range(len(dataset)):
            try:
                query_question_choices = combine_question_choices(
                    dataset[i]["question"], dataset[i]["choices"]
                )
                llm_pred = ori_llm_preds.loc[
                    ori_llm_preds["question_choices"].apply(
                        lambda x: re.sub(r"\s+", "", x)
                    )
                    == re.sub(
                        r"\s+",
                        "",
                        query_question_choices,
                    )
                ]
                if llm_pred.shape[0] == 0:
                    print("no matches")
                    llm_pred = ori_llm_preds.loc[
                        ori_llm_preds["question_choices"].apply(
                            lambda x: fuzz.ratio(
                                re.sub(r"\s+", "", x),
                                re.sub(
                                    r"\s+",
                                    "",
                                    query_question_choices,
                                ),
                            )
                        )
                        >= 90
                    ]
                    print(llm_pred.shape[0])
                if llm_pred.shape[0] > 1:
                    print("many matches!!!")
                    llm_pred = llm_pred.head(1)

                assert llm_pred.shape[0] == 1
            except Exception as e:
                print("llm_pred: ", llm_pred)
                print("question: ", dataset[i]["question"])
                print("question_choices: ", query_question_choices)
                raise e
            # Get majority prediction
            llm_preds = [
                llm_pred[model_name].values[0].strip() for model_name in HELM_RANK
            ]

            gt_agreements = []
            for k in top_ks:
                majority_llm_pred, gt_majority_agreement, gt_ratio_agreement = (
                    get_gt_agreement(dataset[i]["answer"], llm_preds, k)
                )
                gt_agreements += [
                    majority_llm_pred,
                    gt_majority_agreement,
                    gt_ratio_agreement,
                ]

            pred_df.loc[i] = (
                [
                    dataset[i]["question"],
                    dataset[i]["choices"],
                    dataset[i]["answer"],
                    dataset[i]["error_type"],
                ]
                + llm_preds
                + gt_agreements
            )

        pred_df["binary_error_type"] = pred_df["error_type"].apply(
            lambda x: 0 if x == "ok" else 1
        )

        pred_df.to_csv(
            os.path.join(outputs_dir, f"mmlu_multi_experts_{config}.csv"), index=False
        )
        # pred_df = pd.read_csv(
        #     os.path.join(outputs_dir, f"mmlu_multi_experts_{config}.csv")
        # )

        # Calculate performance metrics of the binary classification

        performance_metrics = calculate_performance_metrics(pred_df, top_ks)
        print(performance_metrics)
        plot_metrics_separate(
            performance_metrics,
            config,
            outputs_dir,
        )

        pred_dfs += [pred_df]

    # Combine all the dataframes
    combined_df = pd.concat(pred_dfs)
    combined_df.to_csv(
        os.path.join(outputs_dir, f"mmlu_multi_experts_all.csv"), index=False
    )

    all_performance_metrics = calculate_performance_metrics(combined_df, top_ks)
    plot_metrics_separate(
        all_performance_metrics,
        "all",
        outputs_dir,
    )


if __name__ == "__main__":
    main()
