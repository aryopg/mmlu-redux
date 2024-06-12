import src.corruptions.corruptors as corruptors
from src import constants
from datasets import load_dataset
import numpy
from numpy.random import choice
import os
from dotenv import load_dotenv
import time
from datetime import datetime

load_dotenv(constants.ENV_PATH)


class Corruptor:
    CORRUPTIONS = [
        "wrong_groundtruth",
        "no_correct_answer",
        "multiple_correct_answers",
        "bad_options_clarity",
        "bad_questions_clarity",
    ]

    def __init__(self, probabilities, random_seed, llm=None):
        self._corruption_probabilities = None
        self.corruption_probabilities = probabilities

        self.random_seed = random_seed
        self.llm = self.set_llm(llm, random_seed)

        self.corruption_methods = {
            "wrong_groundtruth": lambda x: corruptors.wrong_grountruth(x),
            "no_correct_answer": lambda x: corruptors.no_correct_answer(x),
            "multiple_correct_answers": lambda x: corruptors.multiple_correct_answers(
                x, self.llm
            ),
            "bad_options_clarity": lambda x: corruptors.bad_options_clarity(x),
            "bad_questions_clarity": lambda x: corruptors.bad_question_clarity(
                x, self.llm
            ),
        }

        self.hf_token = os.getenv("HF_READ_TOKEN")

    @property
    def corruption_probabilities(self):
        return self._corruption_probabilities

    @corruption_probabilities.setter
    def corruption_probabilities(self, probs):
        assert isinstance(
            probs, dict
        ), "Perturbation probabilities should be formatted as a dictionary"
        # if a corruption probability value is missing set it to 0
        self._corruption_probabilities = {
            corr: probs.get(corr, 0) for corr in self.CORRUPTIONS
        }
        total_prob = sum(self.corruption_probabilities.values())

        if total_prob == 0:
            raise ValueError(
                "No corruption probabilities provided. Please specify at least one corruption probability."
            )

        if total_prob > 1:
            raise ValueError(
                "Corruption probabilities sum up to a probability greater than 1. "
                "Please check the probabilities values in the configuration file."
            )

    def set_llm(self, llm, random_seed):
        llm = dict(llm)
        if llm:
            assert isinstance(
                llm, dict
            ), "LLM configuration should be formatted as a dictionary"
            assert (
                "type" in llm
            ), "If you want to use an llm you have to define an llm type in the configuration file"

            if llm["type"] == "http":
                from huggingface_hub import InferenceClient

                client = InferenceClient(**llm["configs"])

                llm["llm_call"] = (
                    lambda x: client.chat_completion(x, seed=random_seed)
                    .choices[0]
                    .message.content
                )

                llm["model"] = llm["configs"]["model"]

            elif llm["type"] == "openai":
                if os.getenv("OPENAI_API_KEY") is None:
                    raise ValueError(
                        "You need to provide an OpenAI key to use an OpenAI model."
                    )
                from openai import OpenAI

                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                completion_config = llm["configs"].get("completion")
                llm["llm_call"] = (
                    lambda x: client.chat.completions.create(
                        messages=x, **completion_config
                    )
                    .choices[0]
                    .message.content
                )
                llm["model"] = completion_config["model"]

        return llm

    def corrupt(self, example):
        """
        Corrupts the given example using a randomly chosen corruption function.

        Parameters:
        example: The example to be corrupted.

        Returns:
            The corrupted example. Modifications depend on the specific corruption function chosen.
        """

        # pick a function to corrupt the example
        # notice that you can simply set the probabilities to 0 to avoid applying a specific corruption
        probabilities = [
            self.corruption_probabilities[corr] for corr in self.CORRUPTIONS
        ]
        corruption_type = choice(self.CORRUPTIONS, replace=False, p=probabilities)
        corruption_function = self.corruption_methods[corruption_type]

        MAX_ATTEMPTS = 5
        SLEEP_PENALTY = 2
        # apply corruption
        for tempts in range(MAX_ATTEMPTS):
            try:
                example = corruption_function(example)
                return example
            except Exception as e:
                print(f"An exeption {e} found for this sample:\n" f"{example}")
                print(
                    f"Trying again. Attempt n. {tempts+1} over {MAX_ATTEMPTS} possible attempts"
                )
                time.sleep(tempts * SLEEP_PENALTY)

        print("Too many attempts. Skipping sample." f"Sample: {example}")
        return None

    def corrupt_dataset(self, dataset_path, dataset_name, output_dir, test_flag=False):
        test_dir = "./test"

        numpy.random.seed(self.random_seed)

        # load clean dataset
        print(f"Loading ***{dataset_name}*** data from", dataset_path)
        dataset_to_corrupt = load_dataset(
            dataset_path, dataset_name, token=self.hf_token
        )

        if test_flag:
            print("Corruption set in test mode")
            dataset_to_corrupt = dataset_to_corrupt["train"].select(
                range(constants.TEST_NUM_SAMPLES)
            )
            output_path = os.path.join(test_dir, "clean_test.csv")
            print("Saving uncorrupted test dataset to ", output_path)
            dataset_to_corrupt.to_csv(output_path)

        corrupted_dataset = dataset_to_corrupt.map(self.corrupt)

        if test_flag:
            # in test scenario saves the result in csv format to facilitate user inspection
            output_path = os.path.join(test_dir, "corruption_test.csv")
            print("Saving corrupted test dataset to ", output_path)
            corrupted_dataset.to_csv(os.path.join(test_dir, "corruption_test.csv"))

        final_output_dir = os.path.join(
            output_dir, f'{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
        )
        print("Saving corrupted dataset to ", final_output_dir)
        corrupted_dataset.save_to_disk(final_output_dir)
