from datasets import load_dataset
import numpy 
from numpy.random import choice
import argparse

global seed 
global client
global llm
global call_llm

"""
We first define the 5 corruption functions, then process the dataset by sampling a corruption function for each example. 
Functions can be sampled with specific probability weights, see the main function for more details.
"""


###########################################################################################################################################
# Wrong groundtruth
# Strategy: randomly select a wrong answer choice and modify the example accordingly.
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



###########################################################################################################################################
# No correct answer
# Strategy: replace the correct answer with 'every option listed'.
# Other strategies (not implemented): remove the correct answer and do nothing.
###########################################################################################################################################


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


###########################################################################################################################################
# Multiple correct answers
# Strategy: generate a new correct answer with the same meaning as the original correct answer. We use an llm for this. 
# The new correct answer is inserted into the list of answer choices. The prompt to generate the new correct answer should be defined 
# in the function generate_answer_with_same_meaning.
# Other strategies (not implemented) : duplicate the correct answer.
###########################################################################################################################################


def generate_answer_with_same_meaning(s):
        # here we should call an llm to edit s
        # duplicate correct answer 
        # or rephrase without changing the semantics
        # for math: use fractions instead of floats and similar
        prompt = {"role": "system", "content": " You are a machine that given a sentence produces another sentence with the same meaning. If the input is just a word, you can return the word."}
        question = {'role': 'user', 'content': f'{s}'}

        return call_llm([prompt, question])


def multiple_correct_answers(example):
    
    example['corruptions'] = 'multiple_correct_answers'
    example['llm for corruption'] = llm
    
    correct_answer = example['choices'][example['answer']]
    
    new_correct_answer = generate_answer_with_same_meaning(correct_answer)

    example['choices'].insert(choice(range(len(example['choices']))) , new_correct_answer)
    example['answer'] = example['choices'].index(correct_answer)

    example['added_correct_answer'] = new_correct_answer

    return example



###########################################################################################################################################
# Bad question clarity
# Strategy: use an llm to generate a new question with the same meaning as the original question.
###########################################################################################################################################

def generate_question_with_same_meaning(s):
        # here we should call an llm to edit s
        # duplicate correct answer 
        # or rephrase without changing the semantics
        # for math: use fractions instead of floats and similar
        prompt = {"role": "system", "content": " Given a question for a multiple choice test, produce another question with the same meaning, but more difficult to understand and somewhat ambiguous."}
        question = {'role': 'user', 'content': f'{s}'}

        return call_llm([prompt, question])

def bad_question_clarity(example):
    example['corruptions'] = 'bad_question_clarity'
    example['llm_for_corruption'] = llm
    
    initial_question = example['question']
    
    bad_question = generate_question_with_same_meaning(initial_question)

    example['question'] = bad_question
    example['original_question'] = initial_question
    
    return example



###########################################################################################################################################
# Bad options clarity
# Strategy: split a false option into 2 options. This is a common corruption in multiple choice questions, 
# where a false option is split into two options during parsing. Here we apply this corruption randomly to one of the false options.
# Other strategies (not implemented): call an LLM to corrupt the options.
###########################################################################################################################################

def bad_options_clarity(example):
    # split a false option into 2 options 
    example['corruptions'] = 'bad_options_clarity'
    correct_answer = example['choices'][example['answer']]

    wrong_answers_idx = set(range(len(example['choices']))) - {example['answer']}
    corrupted_choice_idx = choice(list(wrong_answers_idx))
    corrupted_choice = example['choices'].pop(corrupted_choice_idx)
    
    # split this randomly in the middle
    middle = len(corrupted_choice) // 2
    example['choices'].insert(corrupted_choice_idx, corrupted_choice[:middle])
    example['choices'].insert(corrupted_choice_idx + 1, corrupted_choice[middle:])

    example['answer'] = example['choices'].index(correct_answer)
    
    return example


###########################################################################################################################################
# Corruption process
###########################################################################################################################################


# we sample a corruption function from here
# each corruption function will have a corresponding probability mass
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
    # notice that you can simply set the probabilities to 0 to avoid applying a specific corruption
    corruption_function = choice(CORRUPTION_FUNCTIONS, replace=False, p=corruption_probs)

    # apply corruption
    example = corruption_function(example)

    return example

    

def main(original_dataset_path, corrupted_dataset_dir, hf_token=None, corruption_probs=[0, 0, 0, 0, 0]):

    numpy.random.seed(seed)

    # load clean dataset
    print('Loading ***clean*** data from', original_dataset_path)
    dataset_to_corrupt = load_dataset(original_dataset_path, 'clean', hf_token=hf_token)

    # apply corruptions
    print('Applying corruptions...')
    print('Corruption probabilities:', list(zip(['wrong_grountruth', 'no_correct_answer', 'multiple_correct_answers', 'bad_question_clarity', 'bad_options_clarity'], corruption_probs)))
    corrupted_dataset = dataset_to_corrupt.map(corrupt, fn_kwargs={'corruption_probs': corruption_probs})

    print('Saving corrupted dataset to ', corrupted_dataset_dir)
    corrupted_dataset.save_to_disk(corrupted_dataset_dir)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # general settings
    parser.add_argument('--dataset', help='Hugging Face dataset name to corrupt', default='alessiodevoto/purelabel')
    parser.add_argument('--hf_token', help='Hugging Face token', default=None)
    parser.add_argument('--output_dir', help='Directory to save corrupted dataset', default='./corrupted_dataset')
    parser.add_argument('--seed', help='Random seed', default=42)
    
    # corruption probabilities
    parser.add_argument('--wrong_groundtruth', help='Probability of applying wrong_groundtruth corruption', default=0)
    parser.add_argument('--no_correct_answer', help='Probability of applying no_correct_answer corruption', default=0)
    parser.add_argument('--multiple_correct_answers', help='Probability of applying multiple_correct_answers corruption', default=0)
    parser.add_argument('--bad_question_clarity', help='Probability of applying bad_question_clarity corruption', default=0)
    parser.add_argument('--bad_options_clarity', help='Probability of applying bad_options_clarity corruption', default=0)

    # llm based corruptions
    parser.add_argument('--llm', help='llm to be use for corruption. This can be either a local address for HF TGI (e.g. http://127.0.0.1:8080) or an openai model (e.g. gpt-3.5-turbo)', default=None)
    parser.add_argument('--openai_key', help='OpenAI key', default=False)
    
    
    args = parser.parse_args()

    # set corruption probabilities
    seed = int(args.seed)
    corruption_probs = [float(args.wrong_groundtruth), float(args.no_correct_answer), float(args.multiple_correct_answers), float(args.bad_question_clarity), float(args.bad_options_clarity)]

    if not any(corruption_probs):
        raise ValueError('No corruption probabilities provided. Please specify at least one corruption probability.')

    
    # WARNING: this is very ugly but I am in a rush and it was the easiest way to allow for both huggingface and openai models
    if args.llm:
        print('Make sure you implemented your prompting methods for this llm based corruption.')   
        print('You should define them in the corresponding functions, that are:\n -multiple_correct_answers \n -bad_question_clarity \n -bad_options_clarity')
         
        if 'http' in args.llm:
            from huggingface_hub import InferenceClient
            client = InferenceClient(args.llm)
            def hf_inference(messages):
                return client.chat_completion(messages, seed=seed).choices[0].message.content
            call_llm = hf_inference
            llm = 'llama-3-8b' # TODO change this to the actual model name
        elif 'gpt' in args.llm:
            if not args.openai_key:
                raise ValueError('You need to provide an OpenAI key to use an OpenAI model.')
            from openai import OpenAI
            client = OpenAI(api_key=args.openai_key)
            def openai_inference(messages):
                return client.chat.completions.create(model=args.llm, messages=messages).choices[0].message.content
            call_llm = openai_inference
            llm = args.llm
            

    
    main(args.dataset, args.output_dir, args.hf_token, corruption_probs)


# Example usage for huggingface TGI:
# python corrupt_dataset.py --dataset alessiodevoto/labelchaos --output_dir ./corrupted_dataset --seed 42 --multiple_correct_answers --llm http://127.0.0.1:8080 --hf_token <your_hf_token>

# Example usage for OpenAI:
# python corrupt_dataset.py --dataset alessiodevoto/labelchaos --output_dir ./corrupted_dataset --seed 42 --multiple_correct_answers 1 --llm gpt-3.5-turbo --openai_key <your_openai_key>