import os
from datasets import load_from_disk
from src.constants import ENV_PATH, OUTPUT_DIRECTORY

import argparse

from dotenv import load_dotenv

load_dotenv(ENV_PATH)

perturbations = {
    "BOC": "bad_options_clarity",
    "BQC": "bad_questions_clarity",
    "MCA": "multiple_correct_answers",
    "NCA": "no_correct_answer",
    "WG": "wrong_groundtruth",
}


def push_perturbed_data(perturbation, push):
    directory = os.path.join(OUTPUT_DIRECTORY, "corrupted_data")
    datasets = os.listdir(directory)

    data_paths = [dp for dp in datasets if perturbation in dp]
    assert (
        len(data_paths) == 1
    ), f"there should be at least one dataset and no more than one with perturbation {perturbation}"

    data_path = os.path.join(directory, data_paths[0])

    dataset = load_from_disk(data_path)

    if push:
        dataset.push_to_hub(
            repo_id="edinburgh-dawg/labelchaos",
            config_name=perturbations[args.perturbation],
            token=os.getenv("HF_WRITE_TOKEN"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--perturbation",
        help="Perturbation type. Values accepted: BOC, BQC, MCA, NCA, WG",
        default="BOC",
    )
    parser.add_argument(
        "--push",
        help="If true pushes the dataset on HugginFace",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()

    assert args.perturbation in perturbations.keys(), "unknown perturbation"

    push_perturbed_data(args.perturbation, args.push)
