from datasets import load_dataset, concatenate_datasets, DatasetDict
import sys, os 

import numpy 
numpy.random.seed(42)

from numpy.random import choice

"""
Apply a function to a corrupt dataset such that each example is corrupted in some way.

We first define the corruption functions, then process the dataset by sampling a corruption function. 
Functions can be sampled with specific probability weights, see variables CORRUPTION_FUNCTIONS, CORRUPTION_PROBS.
"""


def wrong_grountruth(example):
    """
    Corrupts an example by randomly selecting a wrong answer choice and modifying the example accordingly.

    Args:
        example: The example to be corrupted. It should have the following keys:
            - 'choices': A list of answer choices.
            - 'answer': The index of the correct answer choice.

    Returns:
        The corrupted example with the following modifications:
            - 'corruptions': A string indicating the type of corruption applied.
            - 'original_grountruth': The original correct answer choice.
            - 'answer': The index of the randomly selected wrong answer choice.
    """
    wrong_answers = set(range(len(example['choices']))) - {example['answer']}
    wrong_groundtruth = choice(list(wrong_answers))

    example['corruptions'] = 'wrong_groundtruth'
    example['original_grountruth'] = example['answer']
    example['answer'] = wrong_groundtruth

    return example


def no_correct_answer(example):
    """
    Removes the correct answer from the given example.

    Args:
        example: The example containing the question and answer choices.

    Returns:
        The modified example with the correct answer removed.
    """
    example['choices'].pop(example['answer'])

    return example


def multiple_correct_answers(example):
    raise NotImplementedError

def bad_question_clarity(example):
    raise NotImplementedError

def bad_options_clarity(example):
    raise NotImplementedError


CORRUPTION_FUNCTIONS = [wrong_grountruth, no_correct_answer, multiple_correct_answers, bad_question_clarity, bad_options_clarity]
CORRUPTION_PROBS = [1/len(CORRUPTION_FUNCTIONS) for i in range(len(CORRUPTION_FUNCTIONS))]


def corrupt(example):
    """
    Corrupts the given example using a randomly chosen corruption function.

    Parameters:
    example: The example to be corrupted.

    Returns:
        The corrupted example. Modifications depend on the specific corruption function chosen.
    """

    # pick a function to corrupt the example
    corruption_function = choice(CORRUPTION_FUNCTIONS, replace=False, p=None)

    # apply corruption
    example = corruption_function(example)

    return example

    

def main(dataset_path):

    # load clean dataset
    dataset_to_corrupt = load_dataset(dataset_path)

    # apply corruptions
    corrupted_dataset = dataset_to_corrupt.map(corrupt)

    corrupted_dataset.save_to_disk(os.path.join(dataset_path, 'corrupted_dataset'))




if __name__ == '__main__':
    main(sys.argv[1])
