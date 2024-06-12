import sys
import os

sys.path.append(os.getcwd())

from src.corruptions.pipeline import Corruptor
from src.constants import (
    CORRUPTED_CONFIGURATION_PATH,
    CORRUPTED_OUTPUT_DIRECTORY,
    RANDOM_SEED,
    DATASET_PATH,
)

import argparse
import yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # general settings
    parser.add_argument(
        "--dataset", help="Hugging Face dataset path to corrupt", default=DATASET_PATH
    )
    parser.add_argument(
        "--name", help="Hugging Face dataset name to corrupt", default="clean"
    )
    parser.add_argument(
        "--output_dir",
        help="Directory to save corrupted dataset",
        default=CORRUPTED_OUTPUT_DIRECTORY,
    )

    with open(CORRUPTED_CONFIGURATION_PATH, "r") as file:
        config = yaml.safe_load(file)

    args = parser.parse_args()

    corruptor = Corruptor(
        probabilities=config["probabilities"],
        random_seed=config.get("seed", RANDOM_SEED),
        llm=config["llm"],
    )

    corruptor.corrupt_dataset(
        dataset_path=args.dataset,
        dataset_name=args.name,
        output_dir=args.output_dir,
        test_flag=config.get("is_a_test", False),
    )
