from datasets import load_dataset
import sys, os 

import numpy 
from numpy.random import choice
import argparse


"""
We first define the 5 corruption functions, then process the dataset by sampling a corruption function for each example. 
Functions can be sampled with specific probability weights, see variables CORRUPTION_FUNCTIONS, CORRUPTION_PROBS.
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
    example['choices'][example['answer']] = 'every option listed'
    example['answer'] = 'n/a'

    return example




def multiple_correct_answers(example):
    
    def generate_answer_with_same_meaning(s):
        # here we should call an llm to edit s
        # duplicate correct answer 
        # or rephrase without changing the semantics
        # for math: use fractions instead of floats and similar
        return s

    example['corruptions'] = 'multiple_correct_answers'
    
    correct_answer = example['choices'][example['answer']]
    
    new_correct_answer = generate_answer_with_same_meaning(correct_answer)

    example['choices'].insert(choice(range(len(example['choices']))) , new_correct_answer)
    example['answer'] = example['choices'].index(correct_answer)

    example['added_correct_answer'] = example['choices'].index(new_correct_answer)

    return example

def bad_question_clarity(example):
    # use mmlu as few shot examples for llm 
    raise NotImplementedError

def bad_options_clarity(example):
    # split a false option into 2 options 

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


def corrupt(example, corruption_probs=[0., 0., 0., 0., 0.]):
    """
    Corrupts the given example using a randomly chosen corruption function.

    Parameters:
    example: The example to be corrupted.

    Returns:
        The corrupted example. Modifications depend on the specific corruption function chosen.
    """

    # pick a function to corrupt the example
    corruption_function = choice(CORRUPTION_FUNCTIONS, replace=False, p=corruption_probs)

    # apply corruption
    example = corruption_function(example)

    return example

    

def main(original_dataset_path, corrupted_dataset_dir, hf_token=None, corruption_probs=[0, 0, 0, 0, 0], seed=42):

    numpy.random.seed(seed)

    # load clean dataset
    print('Loading ***clean*** data from', original_dataset_path)
    dataset_to_corrupt = load_dataset('alessiodevoto/purelabel', 'clean', hf_token=hf_token)

    # apply corruptions
    print('Applying corruptions...')
    print('Corruption probabilities:', list(zip(['wrong_grountruth', 'no_correct_answer', 'multiple_correct_answers', 'bad_question_clarity', 'bad_options_clarity'], corruption_probs)))
    corrupted_dataset = dataset_to_corrupt.map(corrupt, fn_kwargs={'corruption_probs': corruption_probs})

    print('Saving corrupted dataset to ', corrupted_dataset_dir)
    corrupted_dataset.save_to_disk(corrupted_dataset_dir)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', help='Hugging Face dataset name to corrupt', default='alessiodevoto/purelabel')
    parser.add_argument('--hf_token', help='Hugging Face token', default=None)
    parser.add_argument('--output_dir', help='Directory to save corrupted dataset', default='./corrupted_dataset')
    
    parser.add_argument('--wrong_groundtruth', help='Probability of applying wrong_groundtruth corruption', default=0)
    parser.add_argument('--no_correct_answer', help='Probability of applying no_correct_answer corruption', default=0)
    parser.add_argument('--multiple_correct_answers', help='Probability of applying multiple_correct_answers corruption', default=0)
    parser.add_argument('--bad_question_clarity', help='Probability of applying bad_question_clarity corruption', default=0)
    parser.add_argument('--bad_options_clarity', help='Probability of applying bad_options_clarity corruption', default=0)

    parser.add_argument('--openai_key', help='OpenAI API key', default=None)

    parser.add_argument('--seed', help='Random seed', default=42)
    args = parser.parse_args()

    # set corruption probabilities
    corruption_probs = [float(args.wrong_groundtruth), float(args.no_correct_answer), float(args.multiple_correct_answers), float(args.bad_question_clarity), float(args.bad_options_clarity)]

    if not any(corruption_probs):
        raise ValueError('No corruption probabilities provided. Exiting...')
        

    main(args.dataset, args.output_dir, args.hf_token, corruption_probs, args.seed)