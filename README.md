# Don't Send Troops to Stop Ebola: Addressing Groundtruth Errors in MMLU

## Proof-of-Concept

### Run

```bash
# Multi-expert strategy
python scripts/multi_experts.py

# Basic answerability prompt strategy
python scripts/answerability_base.py

# Taxonomy-based answerability prompt strategy
python scripts/answerability_taxonomy.py
python scripts/postprocess_taxonomy_answers.py
```

Things are very much manual and hacky, so you may need to adjust the values within the script instead of argument-based for now.

### Analysis

#### Multi-expert strategy:

I tried to simulate having 3 different LLMs answering the question. Ideally, we would have 3 strong LLMs (e.g., GPT-4, Claude, and Llama 70B). But because i'm stingy, I use GPT-3.5 for this PoC. I prompted GPT-3.5 with temperature = 0, 0.7 and 1 (yes, I could've just prompted it 3 times with non-zero temperature). For this setting, I checked:

The percentage of expert that answers correctly (referred to as "Grountruth-Expert Agreement"): The count of exact match divided by the number of LLMs.
The percentage of agreement between each expert (referred to as "Intra-Expert Agreement"): Pair-wise exact match divided by the number of possible pair-wise interaction.
Aggregate score (Grountruth-Expert Agreement + Intra-Expert Agreement): The sum of the two scores above, such that I can sort with this score.
I didn't do any statistical stuff yet, but from eye test, the Aggregate score looks quite aligned with Joshua's fix.

#### Basic answerability prompt strategy:

I prompted GPT-3.5 to determine whether the ground truth answer is correct or not given the question. The model can only answer "Correct" or "Incorrect". I then simply sort the answer row and do another eye test comparing with Joshua's fix. It seems quite ok, most of the questions that are ambiguous, incorrect, and require expert's opinions are also predicted as "Incorrect" by the model. But there are still some incorrect or ambiguous questions that are predicted as "Correct".

#### Taxonomy-based answerability prompt strategy:

I prompted GPT-3.5 to follow the taxonomy of erroneous question. Starting by checking question presentation, to mc options presentation, then evaluating the answers. This one is quite aligned with Joshua's answers too. But, it only predicts between "Wrong Groundtruth" and "OK". It doesn't predict bad questions or options presentation, which are actually present in the data. Is this just a problem of prompt optimisation?