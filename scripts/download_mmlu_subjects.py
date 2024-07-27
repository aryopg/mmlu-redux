import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv(".env")

import random

import pandas as pd
from datasets import load_dataset


def main():
    random.seed(1234)

    subjects = [
        "college_mathematics",
        "virology",
        "college_chemistry",
        "global_facts",
        "formal_logic",
        "high_school_physics",
        "professional_law",
        "machine_learning",
        "econometrics",
        "public_relations",
    ]
    for subject in subjects:
        dataset = load_dataset("cais/mmlu", subject, split="test")

        df = pd.DataFrame(columns=["question", "choices", "answer", "error_type"])
        random_ids = random.sample(range(len(dataset)), 100)
        for i in range(len(dataset)):
            if i >= 100:
                break
            df.loc[i] = [
                dataset[random_ids[i]]["question"],
                dataset[random_ids[i]]["choices"],
                dataset[random_ids[i]]["answer"],
                "",
            ]

        df.to_csv(f"mmlu_{subject}.csv", index=False)


if __name__ == "__main__":
    main()
