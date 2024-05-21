import json
import pandas as pd
from tqdm import tqdm
from src.taxonomy.data_utils import verbaliser
from sklearn.metrics import f1_score

def compute_metrics(pred_df): 
    exact_match = (pred_df["predicted_error_type"] == pred_df["error_type"]).mean() 
    f1 = f1_score(pred_df["error_type"], pred_df["predicted_error_type"], average="weighted") 
    return {"exact_match": exact_match, "f1_score": f1}

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