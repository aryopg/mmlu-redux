import json
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv(".env")

import pandas as pd


def split_taxonomy_answer(x):
    print(x["prediction"])
    pred = json.loads(x["prediction"].replace("'", '"'))
    x["Question Presentation"] = pred["Question Presentation"]
    x["MC Options Presentation"] = pred["MC Options Presentation"]
    x["Answer Evaluation"] = pred["Answer Evaluation"]
    x["Ground Truth Answer Evaluation"] = pred["Ground Truth Answer Evaluation"]
    x["Classification"] = (
        pred["classification"] if "classification" in pred else pred["Classification"]
    )
    x["Answer"] = pred["answer"]

    return x


df = pd.read_csv("gpt-3.5-turbo_mmlu_answerability_taxonomy_raw.csv")
df = df.apply(split_taxonomy_answer, axis=1)
df.to_csv("gpt-3.5-turbo_mmlu_answerability_taxonomy.csv", index=False)
