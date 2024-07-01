# ARE WE DONE WITH MMLU?
This repository contains the evaluation code for the paper "[**Are We Done With MMLU?**](https://arxiv.org/pdf/2406.04127)"

## MMLU-Redux
MMLU-Redux is a carefully annotated version of the [MMLU (Massive Multitask Language Understanding) dataset](https://arxiv.org/abs/2009.03300) to provide a more accurate and reliable benchmark for evaluating the performance of language models.

## Dataset Overview
MMLU-Redux consists of 30 MMLU subjects, each containing 100 randomly sampled questions.
Please refer to [**🤗 MMLU-Redux Dataset**](https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux) for more details.

## Error Detection Evaluation

This evaluation provides a set of scripts for assessing the error detection capability of various prompting methods on the MMLU-Redux dataset. The methods include Zero-Shot, Zero-Shot with Chain of Thought (CoT), Few-Shot, and Few-Shot with CoT techniques.

### Installation
1. Clone the repository:
```bash
git clone https://github.com/aryopg/mmlu-redux.git
```

2. Navigate to the project directory:
```bash
cd mmlu-redux
```

3. Install open-jdk-21 as a dependency for Pyserini (only for RAG experiments)
```bash
apt-get install openjdk-21-jdk
```

4. Install the required dependencies:
```bash
conda env create -f environment.yml
```


### Usage

#### Zero-Shot Evaluation
To evaluate the Zero-Shot technique on the MMLU-Redux dataset, run the following command:

```bash
python scripts/zero_shot_taxonomy.py
```

To evaluate the Zero-Shot with CoT technique, run:

```bash
python scripts/zero_shot_cot_taxonomy.py
```

#### Few-Shot Evaluation
To evaluate the Few-Shot technique on the MMLU-Redux dataset, run the following command:

```bash
python scripts/few_shot_taxonomy.py
```

To evaluate the Few-Shot with CoT technique, run:

```bash
python scripts/few_shot_cot_taxonomy.py
```

#### RAG Evaluation
To evaluate the RAG on the MMLU-Redux dataset, run the following command:

```bash
python src/retriever/zero_shot_taxonomy_binary.py
```

To evaluate the RAG with CoT technique, run the following command:

```bash
python src/retriever/zero_shot_cot_taxonomy_binary.py
```


#### Evaluating Multiple Datasets
We also provide a convenient bash script to evaluate multiple MMLU-Redux subdatasets using the Chain of Thought (CoT) technique. To run the script, use the following command:

```bash
bash scripts/bash_scripts/mmlu_subdatasets_cot_taxonomy.sh
```

Make sure to modify the script if needed to specify the desired subdatasets and model type.


## Supervised Fine-tuning

### LabelChaos

To validate our fine-tuning strategy for error detection, we developed LabelChaos, a dataset designed to mirror the error distribution of the original MMLU. This dataset serves as a benchmark for finetuning models, which are subsequently evaluated on MMLU-Redux.

To create LabelChaos, we selected and merged six manually labelled datasets. We chose datasets annotated by humans: [OpenBookQA](https://huggingface.co/datasets/allenai/openbookqa), [ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc), [ARC-Easy](https://huggingface.co/datasets/allenai/ai2_arc), [TruthfulQA](https://huggingface.co/datasets/truthfulqa/truthful_qa), [MedQA](https://huggingface.co/datasets/bigbio/med_qa), [MathQA](https://huggingface.co/datasets/allenai/math_qa).

#### Run Setup
For interacting with the HF Hub and/or having access to OpenAI models, create an environment file containing the following keys
```bash
- HF_READ_TOKEN
- HF_WRITE_TOKEN
- OPENAI_API_KEY
```
You can create your own file starting from an [example here](.env_example)

#### Corrupting dataset

With these instructions you can apply perturbations to any dataset structured as MMLU

First, you should define the parameters required for the perturbation.
You should create/edit the configuration file at 'project_dir/corruption/conf.yml'.
You can use [this file](conf/corruption/conf_example.yml) as a reference.

```bash
python scripts/main_corrupt_dataset.py --dataset [DATASET_PATH] --name [DATASET_NAME] --output_dir [A_PATH]
```

where
- [DATASET_PATH] is the HF path of the dataset you want to corrupt. By default, it is 'edinburgh-dawg/labelchaos'
- [DATASET_NAME] is the subset of the dataset you want to corrupt. By default, it is 'clean'
- [A_PATH] a local directory where the output files will be stored. By default, it is 'project_dir/outputs/perturbed'

### Supervised Fine-tuning

We fine-tune the Llama-3 (8B-Instruct) using LabelChaos datasets. To balance the distribution, where most instances are labelled as "__correct__", we adjusted the label distribution to: 0.1 (Wrong Ground Truth), 0.1 (Poor Question Clarity), 0.1 (No Correct Answers), 0.1 (Unclear Options), 0.1 (Multiple Correct Answers), and 0.5 (correct). The training involves 2048 steps, with a batch size of 64, utilizing the AdamW optimizer with a learning rate of 0.0002 and no weight decay. We use LoRA with a rank of 16.
```bash
BATCH=8
GRAD=8
python sft/sft_full.py --model meta-llama/Meta-Llama-3-8B-Instruct --use-bf16 --batch-size ${BATCH} --gradient-accumulation-steps ${GRAD} --max-steps 2048 --collator completion --eval-steps 100 --output-dir lora_binary_uniform_2048 --lora-rank 16
```
