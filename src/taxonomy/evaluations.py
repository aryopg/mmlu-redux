import json

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm

from src.taxonomy.data_utils import verbaliser
from src.taxonomy.model_utils import INSTRUCTION


def compute_metrics(pred_df):
    classes = pred_df["error_type"].unique()
    exact_match = (pred_df["predicted_error_type"] == pred_df["error_type"]).mean()
    cm = confusion_matrix(
        pred_df["error_type"], pred_df["predicted_error_type"], labels=classes
    )

    for i, class_name in enumerate(classes):
        pred_df[f"predicted_{class_name}"] = pred_df["predicted_error_type"].apply(
            lambda x: 1 if x == class_name else 0
        )
        pred_df[f"actual_{class_name}"] = pred_df["error_type"].apply(
            lambda x: 1 if x == class_name else 0
        )

        pred_df[f"{class_name}_TP"] = (
            (pred_df[f"predicted_{class_name}"] == 1)
            & (pred_df[f"actual_{class_name}"] == 1)
        ).astype(int)
        TP = pred_df[f"{class_name}_TP"].sum()

        pred_df[f"{class_name}_FP"] = (
            (pred_df[f"predicted_{class_name}"] == 1)
            & (pred_df[f"actual_{class_name}"] == 0)
        ).astype(int)
        FP = pred_df[f"{class_name}_FP"].sum()

        pred_df[f"{class_name}_FN"] = (
            (pred_df[f"predicted_{class_name}"] == 0)
            & (pred_df[f"actual_{class_name}"] == 1)
        ).astype(int)
        FN = pred_df[f"{class_name}_FN"].sum()

        pred_df[f"{class_name}_TN"] = (
            (pred_df[f"predicted_{class_name}"] == 0)
            & (pred_df[f"actual_{class_name}"] == 0)
        ).astype(int)
        TN = pred_df[f"{class_name}_TN"].sum()

    precision = precision_score(
        pred_df["error_type"], pred_df["predicted_error_type"], average="weighted"
    )
    recall = recall_score(
        pred_df["error_type"], pred_df["predicted_error_type"], average="weighted"
    )
    f1 = f1_score(
        pred_df["error_type"], pred_df["predicted_error_type"], average="weighted"
    )

    metrics = {
        "exact_match": exact_match,
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    return metrics


def compute_metrics_binary(pred_df):
    pred_df["predicted_error_type_binary"] = pred_df["predicted_error_type"].apply(
        lambda x: 1 if x == "ok" else 0
    )
    pred_df["error_type_ok_binary"] = pred_df["error_type_ok"].apply(
        lambda x: 1 if x == "ok" else 0
    )

    pred_df["TP"] = (
        (pred_df["predicted_error_type_binary"] == 1)
        & (pred_df["error_type_ok_binary"] == 1)
    ).astype(int)
    TP = pred_df["TP"].sum()

    pred_df["FP"] = (
        (pred_df["predicted_error_type_binary"] == 1)
        & (pred_df["error_type_ok_binary"] == 0)
    ).astype(int)
    FP = pred_df["FP"].sum()

    pred_df["FN"] = (
        (pred_df["predicted_error_type_binary"] == 0)
        & (pred_df["error_type_ok_binary"] == 1)
    ).astype(int)
    FN = pred_df["FN"].sum()

    pred_df["TN"] = (
        (pred_df["predicted_error_type_binary"] == 0)
        & (pred_df["error_type_ok_binary"] == 0)
    ).astype(int)
    TN = pred_df["TN"].sum()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    exact_match = (
        pred_df["predicted_error_type_binary"] == pred_df["error_type_ok_binary"]
    ).mean()

    return {
        "exact_match": exact_match,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def few_shot_prompt(examples, test_question, test_choices, test_answer):
    messages = [{"role": "system", "content": INSTRUCTION}]

    for example in examples:
        example_prompt = verbaliser(
            example["question"], example["choices"], example["answer"]
        )
        messages.append({"role": "user", "content": example_prompt})
        messages.append(
            {"role": "assistant", "content": json.dumps(example["response"])}
        )

    test_prompt = verbaliser(test_question, test_choices, test_answer)
    messages.append({"role": "user", "content": test_prompt})

    return messages
