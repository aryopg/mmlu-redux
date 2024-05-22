import json
import pandas as pd
from tqdm import tqdm
from src.taxonomy.data_utils import verbaliser
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

import json
import pandas as pd
from tqdm import tqdm
from src.taxonomy.data_utils import verbaliser
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def compute_metrics(pred_df):
    exact_match = (pred_df["predicted_error_type"] == pred_df["error_type"]).mean()

    cm = confusion_matrix(pred_df["error_type"], pred_df["predicted_error_type"])
    
    precision = precision_score(pred_df["error_type"], pred_df["predicted_error_type"], average=None)

    recall = recall_score(pred_df["error_type"], pred_df["predicted_error_type"], average=None)
    
    f1_scores = f1_score(pred_df["error_type"], pred_df["predicted_error_type"], average=None)
    
    macro_precision = precision_score(pred_df["error_type"], pred_df["predicted_error_type"], average="macro")
    macro_recall = recall_score(pred_df["error_type"], pred_df["predicted_error_type"], average="macro")
    macro_f1 = f1_score(pred_df["error_type"], pred_df["predicted_error_type"], average="macro")
    
    micro_precision = precision_score(pred_df["error_type"], pred_df["predicted_error_type"], average="micro")
    micro_recall = recall_score(pred_df["error_type"], pred_df["predicted_error_type"], average="micro")
    micro_f1 = f1_score(pred_df["error_type"], pred_df["predicted_error_type"], average="micro")
    
    return {
        "exact_match": exact_match,
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1_scores": f1_scores,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1
    }

def few_shot_prompt(examples, test_question, test_choices, test_answer):
    example_prompts = []
    for example in examples:
        example_prompt = verbaliser(
            example["question"], example["choices"], example["answer"]
        )
        example_prompts.append(f"{example_prompt}\n{json.dumps(example['response'])}")

    # Add the test question and choices to the prompt
    test_prompt = verbaliser(test_question, test_choices, test_answer)
    example_prompts.append(test_prompt)

    return "\n".join(example_prompts)