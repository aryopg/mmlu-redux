from datasets import load_dataset
import sys, os 

import numpy 
numpy.random.seed(42)
from numpy.random import choice

"""
We first define the 5 corruption functions, then process the dataset by sampling a corruption function for each example. 
Functions can be sampled with specific probability weights, see variables CORRUPTION_FUNCTIONS, CORRUPTION_PROBS.

TODO make sure this is reproducible.
"""


###########################################################################################################################################
# Corruption functions
###########################################################################################################################################

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
    wrong_answers_idx = set(range(len(example['choices']))) - {example['answer']}
    rand_wrong_groundtruth = choice(list(wrong_answers_idx))

    example['corruptions'] = 'wrong_groundtruth'
    example['original_grountruth'] = example['answer']
    example['answer'] = rand_wrong_groundtruth

    return example


def no_correct_answer(example):
    """
    Corrupts the dataset by replacing the correct answer with "all of the above" and setting the answer to 'n/a'.

    Args:
        example (dict): The example dictionary representing a dataset.

    Returns:
        dict: The modified example dictionary with the correct answer replaced and the answer set to 'n/a'.
    """
    
    example['corruptions'] = 'no_correct_answer'
    example['original_correct'] = example['choices'][example['answer']]

    # we replace the correct answer with "all of the above"
    example['choices'][example['answer']] = 'all of the above'
    example['answer'] = 'n/a'

    return example


def multiple_correct_answers(example):
    # duplicate correct answer 
    # or rephrase without changing the semantics
    # for math: use fractions instead of floats and similar
    raise NotImplementedError

def bad_question_clarity(example):
    # use mmlu as few shot examples for llm 
    raise NotImplementedError

def bad_options_clarity(example):
    # split a false option into 2 options

    raise NotImplementedError 

    wrong_answers_idx = set(range(len(example['choices']))) - {example['answer']}
    corrupted_choice_idx = choice(list(wrong_answers_idx))
    corrupted_choice = example['choices'].pop(corrupted_choice_idx)
    
    # split this randomly in the middle
    
    example['corruptions'] = 'bad_options_clarity'
    example['original_grountruth'] = example['answer']
    example['answer'] = corrupted_choice
    raise NotImplementedError


###########################################################################################################################################
# Corruption process
###########################################################################################################################################


# we sample a corruption function from here 
# each corruption function has a corresponding probability mass
CORRUPTION_FUNCTIONS = [wrong_grountruth, no_correct_answer, multiple_correct_answers, bad_question_clarity, bad_options_clarity]
CORRUPTION_PROBS = [1, 0, 0, 0, 0] #[1/len(CORRUPTION_FUNCTIONS) for _ in range(len(CORRUPTION_FUNCTIONS))]


def corrupt(example):
    """
    Corrupts the given example using a randomly chosen corruption function.

    Parameters:
    example: The example to be corrupted.

    Returns:
        The corrupted example. Modifications depend on the specific corruption function chosen.
    """

    # pick a function to corrupt the example
    corruption_function = choice(CORRUPTION_FUNCTIONS, replace=False, p=CORRUPTION_PROBS)

    # apply corruption
    example = corruption_function(example)

    return example

    

def main(original_dataset_path, corrupted_dataset_dir, hf_token=None):

    # load clean dataset
    print('Loading ***clean*** data from', original_dataset_path)
    dataset_to_corrupt = load_dataset(original_dataset_path, hf_token=hf_token, split='clean')

    # apply corruptions
    corrupted_dataset = dataset_to_corrupt.map(corrupt)

    print('Saving corrupted dataset to ', corrupted_dataset_dir)
    corrupted_dataset.save_to_disk(corrupted_dataset_dir)






if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python corrupt_dataset.py <path_to_original_data> <path_to_output_dir> <hf_token_optional>')
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
