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
    fbeta_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

load_dotenv(dotenv_path=".env")

HELM_RANK_ADJ = {
    "college_chemistry" : [
        "openai_gpt-4o-2024-05-13",
        "mistralai_mixtral-8x22b",
        "anthropic_claude-3-opus-20240229",
        "openai_gpt-4-0613",
        "meta_llama-3-70b",
        "writer_palmyra-x-v3",
        "google_text-unicorn@001",
        "google_gemini-1.5-pro-preview-0409",
        "google_gemini-1.5-flash-preview-0514",
        "openai_gpt-4-1106-preview",
    ],
    "college_mathematics": [
        "openai_gpt-4-0613",
        "google_gemini-1.5-pro-preview-0409",
        "meta_llama-3-70b",
        "google_gemini-1.5-flash-preview-0514",
        "google_text-unicorn@001",
        "anthropic_claude-3-opus-20240229",
        "writer_palmyra-x-v3",
        "openai_gpt-4o-2024-05-13",
        "mistralai_mixtral-8x22b",
        "openai_gpt-4-1106-preview",
    ],
    "econometrics" : [
        "anthropic_claude-3-opus-20240229",
        "google_gemini-1.5-pro-preview-0409",
        "openai_gpt-4o-2024-05-13",
        "openai_gpt-4-1106-preview",
        "mistralai_mixtral-8x22b",
        "openai_gpt-4-0613",
        "meta_llama-3-70b",
        "writer_palmyra-x-v3",
        "google_text-unicorn@001",
        "google_gemini-1.5-flash-preview-0514",
    ],
    "formal_logic" : [
        "anthropic_claude-3-opus-20240229",
        "google_gemini-1.5-pro-preview-0409",
        "google_text-unicorn@001",
        "writer_palmyra-x-v3",
        "google_gemini-1.5-flash-preview-0514",
        "mistralai_mixtral-8x22b",
        "meta_llama-3-70b",
        "openai_gpt-4-0613",
        "openai_gpt-4o-2024-05-13",
        "openai_gpt-4-1106-preview",
    ],
    "global_facts" : [
        "anthropic_claude-3-opus-20240229",
        "google_gemini-1.5-pro-preview-0409",
        "openai_gpt-4-1106-preview",
        "openai_gpt-4o-2024-05-13",
        "google_gemini-1.5-flash-preview-0514",
        "openai_gpt-4-0613",
        "google_text-unicorn@001",
        "writer_palmyra-x-v3",
        "mistralai_mixtral-8x22b",
        "meta_llama-3-70b",
    ],
    "high_school_physics" : [
        "anthropic_claude-3-opus-20240229",
        "openai_gpt-4o-2024-05-13",
        "openai_gpt-4-0613",
        "google_gemini-1.5-flash-preview-0514",
        "google_gemini-1.5-pro-preview-0409",
        "openai_gpt-4-1106-preview",
        "meta_llama-3-70b",
        "writer_palmyra-x-v3",
        "google_text-unicorn@001",
        "mistralai_mixtral-8x22b",
    ],
    "machine_learning" : [
        "anthropic_claude-3-opus-20240229",
        "openai_gpt-4-0613",
        "meta_llama-3-70b",
        "openai_gpt-4-1106-preview",
        "openai_gpt-4o-2024-05-13",
        "mistralai_mixtral-8x22b",
        "google_text-unicorn@001",
        "writer_palmyra-x-v3",
        "google_gemini-1.5-pro-preview-0409",
        "google_gemini-1.5-flash-preview-0514",
    ],
    "professional_law" : [
        "openai_gpt-4o-2024-05-13",
        "openai_gpt-4-0613",
        "openai_gpt-4-1106-preview",
        "anthropic_claude-3-opus-20240229",
        "google_gemini-1.5-pro-preview-0409",
        "google_text-unicorn@001",
        "writer_palmyra-x-v3",
        "meta_llama-3-70b",
        "google_gemini-1.5-flash-preview-0514",
        "mistralai_mixtral-8x22b",
    ],
    "public_relations" : [
        "anthropic_claude-3-opus-20240229",
        "openai_gpt-4-1106-preview",
        "mistralai_mixtral-8x22b",
        "openai_gpt-4o-2024-05-13",
        "writer_palmyra-x-v3",
        "google_text-unicorn@001",
        "google_gemini-1.5-flash-preview-0514",
        "google_gemini-1.5-pro-preview-0409",
        "meta_llama-3-70b",
        "openai_gpt-4-0613",
    ],
    "virology" : [
        "openai_gpt-4o-2024-05-13",
        "openai_gpt-4-0613",
        "openai_gpt-4-1106-preview",
        "meta_llama-3-70b",
        "mistralai_mixtral-8x22b",
        "anthropic_claude-3-opus-20240229",
        "google_gemini-1.5-pro-preview-0409",
        "writer_palmyra-x-v3",
        "google_text-unicorn@001",
        "google_gemini-1.5-flash-preview-0514",
    ],
}


# HELM_RANK = [
#     "anthropic_claude-3-opus-20240229",
#     "openai_gpt-4o-2024-05-13",
#     "openai_gpt-4-0613",
#     "google_gemini-1.5-pro-preview-0409",
#     "openai_gpt-4-1106-preview",
#     "meta_llama-3-70b",
#     "writer_palmyra-x-v3",
#     "google_text-unicorn@001",
#     "mistralai_mixtral-8x22b",
#     "google_gemini-1.5-flash-preview-0514",
# ]

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


def calculate_normalised_entropy(predictions):
    # Calculate total annotations
    total_annotations = sum(predictions)

    # If there are no annotations, entropy is zero (fully consistent)
    if total_annotations == 0:
        return 0.0

    # Calculate proportions
    proportions = [count / total_annotations for count in predictions]

    # Calculate entropy
    entropy = -sum(p * math.log2(p) for p in proportions if p > 0)

    # Calculate maximum entropy - log2(num_labels) = log2(4) = 2
    max_entropy = 2

    # Calculate normalised entropy
    normalised_entropy = entropy / max_entropy

    return normalised_entropy


def get_gt_agreement(gt_answer, llm_preds, top_k):
    subset_llm_preds = llm_preds[:top_k]
    majority_llm_pred = max(set(subset_llm_preds), key=subset_llm_preds.count)
    gt_majority_agreement = chr(65 + gt_answer) == majority_llm_pred.strip()
    gt_ratio_agreement = sum(
        [1 for pred in subset_llm_preds if pred != chr(65 + gt_answer)]
    ) / len(subset_llm_preds)

    # Entropy intra-experts
    # Initialize a dictionary with possible labels as keys and counts as values
    possible_labels = ["A", "B", "C", "D"]
    label_counts = {label: 0 for label in possible_labels}

    # Count occurrences of each label in the list
    for label in subset_llm_preds:
        if label.strip() not in label_counts:
            #print(f"{label} is not a valid answer.")
            continue
        label_counts[label.strip()] += 1

    # Create the data row based on the counts
    data_row = [label_counts[label] for label in possible_labels]
    normalised_entropy = calculate_normalised_entropy(data_row)

    return (
        majority_llm_pred,
        gt_majority_agreement,
        gt_ratio_agreement,
        normalised_entropy,
    )


def calculate_performance_metrics(pred_df, top_ks):
    gt_majority_agreement_tps = {k: 0 for k in top_ks}
    gt_majority_agreement_tns = {k: 0 for k in top_ks}
    gt_majority_agreement_fns = {k: 0 for k in top_ks}
    gt_majority_agreement_fps = {k: 0 for k in top_ks}
    gt_majority_agreement_accs = {k: 0 for k in top_ks}
    gt_majority_agreement_precisions = {k: 0 for k in top_ks}
    gt_majority_agreement_recalls = {k: 0 for k in top_ks}
    gt_majority_agreement_f1_scores = {k: 0 for k in top_ks}
    gt_majority_agreement_f2_scores = {k: 0 for k in top_ks}
    gt_ratio_agreement_tps = {k: 0 for k in top_ks}
    gt_ratio_agreement_tns = {k: 0 for k in top_ks}
    gt_ratio_agreement_fns = {k: 0 for k in top_ks}
    gt_ratio_agreement_fps = {k: 0 for k in top_ks}
    gt_ratio_agreement_accs = {k: 0 for k in top_ks}
    gt_ratio_agreement_precisions = {k: 0 for k in top_ks}
    gt_ratio_agreement_recalls = {k: 0 for k in top_ks}
    gt_ratio_agreement_f1_scores = {k: 0 for k in top_ks}
    gt_ratio_agreement_f2_scores = {k: 0 for k in top_ks}
    # gt_ratio_agreement_auprcs = {k: 0 for k in top_ks}
    entropy_tps = {k: 0 for k in top_ks}
    entropy_tns = {k: 0 for k in top_ks}
    entropy_fns = {k: 0 for k in top_ks}
    entropy_fps = {k: 0 for k in top_ks}
    entropy_accs = {k: 0 for k in top_ks}
    entropy_precisions = {k: 0 for k in top_ks}
    entropy_recalls = {k: 0 for k in top_ks}
    entropy_f1_scores = {k: 0 for k in top_ks}
    entropy_f2_scores = {k: 0 for k in top_ks}
    ratio_w_entropy_tps = {k: 0 for k in top_ks}
    ratio_w_entropy_tns = {k: 0 for k in top_ks}
    ratio_w_entropy_fns = {k: 0 for k in top_ks}
    ratio_w_entropy_fps = {k: 0 for k in top_ks}
    ratio_w_entropy_accs = {k: 0 for k in top_ks}
    ratio_w_entropy_precisions = {k: 0 for k in top_ks}
    ratio_w_entropy_recalls = {k: 0 for k in top_ks}
    ratio_w_entropy_f1_scores = {k: 0 for k in top_ks}
    ratio_w_entropy_f2_scores = {k: 0 for k in top_ks}

    for k in top_ks:
        pred_df[f"pred_gt_majority_agreement_top_{k}"] = pred_df[
            f"gt_majority_agreement_top_{k}"
        ].apply(lambda x: 0 if x else 1)

        pred_df[f"pred_gt_ratio_agreement_top_{k}"] = pred_df[
            f"gt_ratio_agreement_top_{k}"
        ].apply(lambda x: 0 if x < 0.5 else 1)

        pred_df[f"pred_entropy_top_{k}"] = pred_df[f"entropy_top_{k}"].apply(
            lambda x: 0 if x < 0.5 else 1
        )

        pred_df[f"pred_ratio_w_entropy_top_{k}"] = pred_df[
            [f"gt_ratio_agreement_top_{k}", f"entropy_top_{k}"]
        ].apply(
            lambda x: (
                0
                if x[f"gt_ratio_agreement_top_{k}"] - x[f"entropy_top_{k}"] < 0.5
                else 1
            ),
            axis=1,
        )

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
        gt_majority_agreement_f2_scores[k] = fbeta_score(
            pred_df["binary_error_type"],
            pred_df[f"pred_gt_majority_agreement_top_{k}"],
            beta=2,
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
        gt_ratio_agreement_f2_scores[k] = fbeta_score(
            pred_df["binary_error_type"],
            pred_df[f"pred_gt_ratio_agreement_top_{k}"],
            beta=2,
        )
        # gt_ratio_agreement_auprcs[k] = average_precision_score(
        #     pred_df["binary_error_type"], pred_df[f"gt_ratio_agreement_top_{k}"]
        # )

        (tn, fp), (fn, tp) = confusion_matrix(
            pred_df["binary_error_type"], pred_df[f"pred_entropy_top_{k}"]
        )
        entropy_tps[k] = tp
        entropy_tns[k] = tn
        entropy_fns[k] = fn
        entropy_fps[k] = fp
        entropy_accs[k] = accuracy_score(
            pred_df["binary_error_type"], pred_df[f"pred_entropy_top_{k}"]
        )
        entropy_precisions[k] = precision_score(
            pred_df["binary_error_type"], pred_df[f"pred_entropy_top_{k}"]
        )
        entropy_recalls[k] = recall_score(
            pred_df["binary_error_type"], pred_df[f"pred_entropy_top_{k}"]
        )
        entropy_f1_scores[k] = f1_score(
            pred_df["binary_error_type"], pred_df[f"pred_entropy_top_{k}"]
        )
        entropy_f2_scores[k] = fbeta_score(
            pred_df["binary_error_type"], pred_df[f"pred_entropy_top_{k}"], beta=2
        )

        (tn, fp), (fn, tp) = confusion_matrix(
            pred_df["binary_error_type"], pred_df[f"pred_entropy_top_{k}"]
        )
        ratio_w_entropy_tps[k] = tp
        ratio_w_entropy_tns[k] = tn
        ratio_w_entropy_fns[k] = fn
        ratio_w_entropy_fps[k] = fp
        ratio_w_entropy_accs[k] = accuracy_score(
            pred_df["binary_error_type"], pred_df[f"pred_ratio_w_entropy_top_{k}"]
        )
        ratio_w_entropy_precisions[k] = precision_score(
            pred_df["binary_error_type"], pred_df[f"pred_ratio_w_entropy_top_{k}"]
        )
        ratio_w_entropy_recalls[k] = recall_score(
            pred_df["binary_error_type"], pred_df[f"pred_ratio_w_entropy_top_{k}"]
        )
        ratio_w_entropy_f1_scores[k] = f1_score(
            pred_df["binary_error_type"], pred_df[f"pred_ratio_w_entropy_top_{k}"]
        )
        ratio_w_entropy_f2_scores[k] = fbeta_score(
            pred_df["binary_error_type"],
            pred_df[f"pred_ratio_w_entropy_top_{k}"],
            beta=2,
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
        "Majority Agreement F2 Score": gt_majority_agreement_f2_scores,
        "Ratio Disagreement tp": gt_ratio_agreement_tps,
        "Ratio Disagreement tn": gt_ratio_agreement_tns,
        "Ratio Disagreement fn": gt_ratio_agreement_fns,
        "Ratio Disagreement fp": gt_ratio_agreement_fps,
        "Ratio Disagreement Accuracy": gt_ratio_agreement_accs,
        "Ratio Disagreement Precision": gt_ratio_agreement_precisions,
        "Ratio Disagreement Recall": gt_ratio_agreement_recalls,
        "Ratio Disagreement F1 Score": gt_ratio_agreement_f1_scores,
        "Ratio Disagreement F2 Score": gt_ratio_agreement_f2_scores,
        "Entropy tp": entropy_tps,
        "Entropy tn": entropy_tns,
        "Entropy fn": entropy_fns,
        "Entropy fp": entropy_fps,
        "Entropy Accuracy": entropy_accs,
        "Entropy Precision": entropy_precisions,
        "Entropy Recall": entropy_recalls,
        "Entropy F1 Score": entropy_f1_scores,
        "Entropy F2 Score": entropy_f2_scores,
        "Ratio w/ Entropy tp": ratio_w_entropy_tps,
        "Ratio w/ Entropy tn": ratio_w_entropy_tns,
        "Ratio w/ Entropy fn": ratio_w_entropy_fns,
        "Ratio w/ Entropy fp": ratio_w_entropy_fps,
        "Ratio w/ Entropy Accuracy": ratio_w_entropy_accs,
        "Ratio w/ Entropy Precision": ratio_w_entropy_precisions,
        "Ratio w/ Entropy Recall": ratio_w_entropy_recalls,
        "Ratio w/ Entropy F1 Score": ratio_w_entropy_f1_scores,
        "Ratio w/ Entropy F2 Score": ratio_w_entropy_f2_scores,
        # "Ratio Agreement AUPRC": gt_ratio_agreement_auprcs,
    }


def plot_metrics_separate(data_dicts, config, outputs_dir):
    num_plots = len(data_dicts)
    fig, axs = plt.subplots(4, int(num_plots / 4), figsize=(5 * int(num_plots / 4), 24))

    for idx, (label, data) in enumerate(data_dicts.items()):
        keys = list(data.keys())
        values = list(data.values())
        axs[math.floor(idx / (num_plots / 4))][int(idx % (num_plots / 4))].plot(
            keys, values, marker="o"
        )
        axs[math.floor(idx / (num_plots / 4))][int(idx % (num_plots / 4))].set_title(
            label
        )
        axs[math.floor(idx / (num_plots / 4))][int(idx % (num_plots / 4))].grid(True)
        # axs[math.floor(idx / (num_plots / 4))][idx].set_ylabel()

    axs[0][-1].set_xlabel("Top K")
    axs[1][-1].set_xlabel("Top K")
    # plt.tight_layout()
    # plt.show()
    fig.savefig(
        os.path.join(outputs_dir, f"persub_adj_mmlu_multi_experts_{config}_performance.png")
    )


def main():
    pred_dfs = []
    for config in CONFIGS:
        print(f"Processing Mini MMLU {config}")
        outputs_dir = "./outputs/multi_expert_helm/"
        # top_ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        top_ks = [3, 5, 7, 9, 10]

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
                f"entropy_top_{k}",
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
                llm_pred[model_name].values[0].strip() for model_name in HELM_RANK_ADJ[config]
            ]

            gt_agreements = []
            for k in top_ks:
                (
                    majority_llm_pred,
                    gt_majority_agreement,
                    gt_ratio_agreement,
                    entropy,
                ) = get_gt_agreement(dataset[i]["answer"], llm_preds, k)
                gt_agreements += [
                    majority_llm_pred,
                    gt_majority_agreement,
                    gt_ratio_agreement,
                    entropy,
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
            os.path.join(outputs_dir, f"persub_adj_mmlu_multi_experts_{config}.csv"), index=False
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
        os.path.join(outputs_dir, f"persub_adj_mmlu_multi_experts_all.csv"), index=False
    )

    all_performance_metrics = calculate_performance_metrics(combined_df, top_ks)
    plot_metrics_separate(
        all_performance_metrics,
        "all",
        outputs_dir,
    )


if __name__ == "__main__":
    main()
