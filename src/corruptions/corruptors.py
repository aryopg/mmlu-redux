from numpy.random import choice

TEST_NUM_SAMPLES = 100


def check_llm_already_defined(llm):
    if isinstance(llm, dict):
        if 'type' in llm and 'model' in llm:
            return True
        else:
            raise ValueError('llm global variable is missing \'type\' and \'model\' fields')
    else:
        raise ValueError('llm global variable must be defined with llm parameters')


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


def generate_answer_with_same_meaning(s, llm):
    # here we should call an llm to edit s
    # duplicate correct answer
    # or rephrase without changing the semantics
    # for math: use fractions instead of floats and similar
    prompt = {"role": "system",
              "content": " You are a machine that given a sentence produces another sentence with the same meaning. If the input is just a word, you can return the word."}
    question = {'role': 'user', 'content': f'{s}'}

    return llm['llm_call']([prompt, question])


def multiple_correct_answers(example, llm):
    check_llm_already_defined(llm)

    example['corruptions'] = 'multiple_correct_answers'
    example['llm for corruption'] = llm['model']

    correct_answer = example['choices'][example['answer']]

    new_correct_answer = generate_answer_with_same_meaning(correct_answer, llm)

    example['choices'].insert(choice(range(len(example['choices']))), new_correct_answer)
    example['answer'] = example['choices'].index(correct_answer)

    example['added_correct_answer'] = new_correct_answer

    return example


###########################################################################################################################################
# Bad question clarity
# Strategy: use an llm to generate a new question with the same meaning as the original question.
###########################################################################################################################################

def generate_question_with_same_meaning(s, llm):
    # here we should call an llm to edit s
    # duplicate correct answer
    # or rephrase without changing the semantics
    # for math: use fractions instead of floats and similar

    # four different prompts for building less clear questions.
    # the first is the most general one.
    # the second removes context information from the question
    # the third adds a reference to external content
    # the last adds a reference to another question
    prompts = [
        {"role": "system",
         "content": "You are a machine that given a question from a multiple-choice question produces another question "
                    "with the same meaning but it is less clear. "
                    "The outcome must be meaningful but less clear than the input."},
        {"role": "system",
         "content": "You are a machine that given a question from a multiple-choice question produces another question"
                    " with the same meaning but less clear."
                    "You make it less clear by removing context information that is necessary for answering properly."
                    "The outcome must be meaningful but less clear than the input."},
        {"role": "system",
         "content": "You are a machine that given a question from a multiple-choice question produces another question"
                    " with the same meaning but less clear."
                    "You make it less clear by adding a reference to external content."
                    "The outcome must be meaningful but less clear than the input."},
        {"role": "system",
         "content": "You are a machine that given a question from a multiple-choice question produces another question"
                    " with the same meaning but less clear."
                    "You make it less clear by adding a reference to another question indicated by its number."
                    "The outcome must be meaningful but less clear than the input."}
    ]

    prompt = choice(prompts, p=[0.1, 0.3, 0.3, 0.3])

    question = {'role': 'user', 'content': f'{s}'}

    return llm['llm_call']([prompt, question])


def bad_question_clarity(example, llm):
    check_llm_already_defined(llm)

    example['corruptions'] = 'bad_question_clarity'
    example['llm_for_corruption'] = llm['model']

    initial_question = example['question']

    bad_question = generate_question_with_same_meaning(initial_question, llm)

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
