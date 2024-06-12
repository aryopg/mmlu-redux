import re
import string


CHOICES_DELIMITER = "\n"
QUESTION_VERBALISER = "Question: {question}\nChoices:\n{choices}\nGround Truth Answer: {answer}\nYour Response: "


def verbaliser(question, choices, answer):
    verbalised_choices = CHOICES_DELIMITER.join(
        [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)]
    )
    return QUESTION_VERBALISER.format(
        question=question,
        choices=verbalised_choices,
        answer=f"{chr(65+answer)}. {choices[answer]}",
    )


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_error_type(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def extract_braced_content(text):
    match = re.search(r"\{.*?\}", text)
    if match:
        return match.group(0)
    else:
        return ""
