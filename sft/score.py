from datasets import load_dataset, interleave_datasets, concatenate_datasets
import transformers
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import sys
from tqdm import tqdm
import json

from utils import NestedKeyDataset
import evaluate

from collections import Counter

from sklearn.metrics import confusion_matrix


def main():

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    trues = []
    preds = []

    preds_full = []
    trues_full = []

    subset_lst = [
        "bad_options_clarity",
        "bad_question_clarity",
        "clean",
        "multiple_correct_answers",
        "no_correct_answer",
        "wrong_groundtruth",
    ]
    # subset_lst = ['bad_options_clarity', 'bad_question_clarity', 'clean', 'multiple_correct_answers', 'no_correct_answer', 'wrong_groundtruth']
    label_to_id = {label.replace("_", " "): id for id, label in enumerate(subset_lst)}
    print(label_to_id)

    remapping = {
        "ok": 1,
        "clean": 1,
        "bad_question_clarity".replace("_", " "): 2,
        "bad_options_clarity".replace("_", " "): 2,
        "bad presentation".replace("_", " "): 2,
        "no_correct_answer".replace("_", " "): 3,
        "multiple_correct_answers".replace("_", " "): 3,
        "wrong_groundtruth".replace("_", " "): 3,
        "Here is the truth": 1,
        "A. Valid": 1,
        "Valid": 1,
        "Here is the complete": 1,
        "Let's construct the": 1,
        "Invalid. Counterexample": 3,
        "First, let's": 1,
        "D. (i": 1,
        "A. Valid": 1,
        "B. (i": 1,
        "conclusion": 1,
    }

    # with open("mmlu_full.jsonl") as reader:
    # with open("mmlu_full_cat.jsonl") as reader:
    # with open("mmlu_full_cat_uniform_train.jsonl") as reader:
    with open("mmlu_full_cat_uniform_2048.jsonl") as reader:
        for line in reader:
            if line.startswith('{"input"'):
                items = json.loads(line.strip())
                true_label = items["input"]["messages"][-1]["content"]
                pred_label = items["pred"].replace("questions", "question")
                # print(true_label, pred_label)

                # trues.append(1 if "yes" in true_label else 0)
                # preds.append(1 if "yes" in pred_label else 0)
                # preds.append(label_to_id[pred_label])
                # trues.append(label_to_id["clean"] if true_label == "ok" else label_to_id[true_label])

                preds.append(0 if remapping[pred_label] == 1 else 1)
                trues.append(0 if remapping[true_label] == 1 else 1)
                # preds.append(remapping[pred_label])
                # trues.append(remapping[true_label])
                # preds.append(pred_label)
                # trues.append(true_label)
                # preds_full.append(pred_label)
                # preds_full.append(pred_label if pred_label == "clean" else "wrong")
                # trues_full.append("clean" if true_label == "ok" else true_label)
                # trues_full.append("clean" if true_label == "ok" else "wrong")
                # break

    # print(Counter(preds_full))
    # print(Counter(trues_full))

    print(Counter(preds))
    print(Counter(trues))
    print(accuracy_metric.compute(references=trues, predictions=preds))
    print(f1_metric.compute(references=trues, predictions=preds))
    # print(f1_metric.compute(references=trues, predictions=preds, average="macro"))
    # print(f1_metric.compute(references=trues, predictions=preds, average="micro"))
    # print(confusion_matrix(trues_full, preds_full, labels=[label.replace("_", " ") for label in subset_lst]))
    # print(confusion_matrix(trues_full, preds_full, labels=["clean", "wrong"]).ravel())
    print(confusion_matrix(trues, preds))
    # tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    # print(set(preds))
    # print(set(trues))


if __name__ == "__main__":
    main()
