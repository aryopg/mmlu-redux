#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import argparse
import logging
import re
import wandb

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from transformers.utils import is_bitsandbytes_available, is_flash_attn_2_available

from datasets import load_dataset, interleave_datasets

from trl import SFTTrainer, setup_chat_format, DataCollatorForCompletionOnlyLM
from peft import LoraConfig


from utils import (
    check_bf16_support,
    smart_tokenizer_and_embedding_resize,
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_PROJECT"] = "mmlu-llm_v1"

logger = logging.getLogger(name=__file__)
logger.setLevel(logging.INFO)

# For debugging/prototyping:
# PYTHONPATH=. ./cli/mt-sft-cli.py --use-bf16 --learning-rate 1e-4 --batch-size 4 --collator completion --gradient-accumulation-steps 1 --max-steps 8 --model google/gemma-2b --lora-rank 8 --warmup-steps 4 --max-seq-length 4096

DEFAULT_PAD_TOKEN = "[PAD]"


def create_model_path(args):
    # Sanitize the model ID to replace '/' with '_' to avoid directory path issues
    sanitized_model_id = re.sub(r"[/\\]", "_", args.model.split("/")[-1])

    path_parts = [
        sanitized_model_id,
        f"bs={args.batch_size}",
        f"lr={args.learning_rate:.0e}",
        f"collator={args.collator}",
        f"gas={args.gradient_accumulation_steps}",
        f"ms={args.max_steps}",
        f"lrank={args.lora_rank}",
        f"ws={args.warmup_steps}",
    ]

    # Combine all parts with underscores
    path = "_".join(path_parts)

    # Sanitise, due to the following:
    # huggingface_hub.utils._validators.HFValidationError: Repo id must [..]
    # path = path.replace('_', '-')
    path = path.replace("=", "_")

    return path


def main(argv):
    parser = argparse.ArgumentParser(description="Note-to-FHIR Trainer Model")

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Base model ID",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="outputs",
        help="Output directory for training artifacts",
    )
    parser.add_argument(
        "--use-bf16", "--bf16", action="store_true", help="BF16 support if available"
    )
    parser.add_argument(
        "--use-quantization",
        "-q",
        action="store_true",
        help="Enable quantization with BitsAndBytes",
    )

    parser.add_argument("--optim", type=str, default="adamw_torch", help="Optimizer")
    parser.add_argument("--eval-steps", type=int, default=8, help="Evaluation steps")
    parser.add_argument(
        "--learning-rate", "--lr", type=float, default=2e-4, help="Learning rate"
    )
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--gradient-accumulation-steps",
        "-g",
        type=int,
        default=8,
        help="No. of gradient accumulation steps",
    )
    parser.add_argument(
        "--warmup-steps", "-w", type=int, default=100, help="No. of warmup steps"
    )

    parser.add_argument(
        "--max-steps", type=int, default=1024, help="Max no. of training steps"
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=4096, help="Max sequence length"
    )

    parser.add_argument(
        "--collator",
        action="store",
        type=str,
        default="completion",
        choices=["completion", "default"],
    )
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")

    # Parse arguments
    args = parser.parse_args()

    use_bf16 = check_bf16_support(args.use_bf16)

    model_id = args.model

    bnb_config = None
    if args.use_quantization and is_bitsandbytes_available():
        logger.info("Using quantisation ..")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        )
        model_kwargs = {"quantization_config": bnb_config}
    else:
        model_kwargs = {"torch_dtype": torch.bfloat16} if use_bf16 else {}

    if is_flash_attn_2_available() and use_bf16:
        logger.info("Using Flash Attention ..")
        model_kwargs.update({"attn_implementation": "flash_attention_2"})

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_cache=False,
        **model_kwargs,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")

    ds_id = "edinburgh-dawg/labelchaos"

    subset_lst = [
        "bad_options_clarity",
        "bad_questions_clarity",
        "clean",
        "multiple_correct_answers",
        "no_correct_answer",
        "wrong_groundtruth",
    ]
    subset_id_to_ds = {k: load_dataset(ds_id, k) for k in subset_lst}

    DEFAULT_INSTRUCTION = "Analyze the following multiple-choice question and the corresponding answer carefully, and tell me which category it falls in:\n\n1. bad options clarity\n2. bad questions clarity\n3. clean\n4. multiple correct answers\n5. no correct answer\n6. wrong groundtruth"
    # DEFAULT_INSTRUCTION = "Analyze the following multiple-choice question and the corresponding answer carefully, and tell me which category it falls in:\n\n1. bad presentation\n2. clean\n3. wrong groundtruth"
    DEFAULT_INSTRUCTION = (
        "# Task:\n"
        "Given a triple consisting of a multiple choice question, its choices, and the corresponding ground truth answer, your task is to classify the triple into 'ok' or 'not ok'.\n\n"
        "# Instructions:\n"
        "1. Question Presentation: Is the question well-presented? Assess clarity, grammar, and sufficiency of information.\n"
        "1.1 If Yes, assess the presentation of the MC options.\n"
        "1.2 If No, classify the issue as 'not ok'.\n"
        "2. MC Options Presentation: Are the MC options well-presented? Check if the options are clear, distinct, and relevant to the question.\n"
        "2.1 If Yes, determine if there is one potentially correct answer.\n"
        "2.2 If No, classify the issue as 'not ok'.\n"
        "3. Answer Evaluation: Is there one, more than one, or no potentially correct answer in the options list?\n"
        "3.1 If one, continue to Ground Truth Answer Evaluation.\n"
        "3.2 If more than one, classify the issue as 'not ok'.\n"
        "3.3 If no correct answer, classify the issue as 'not ok'.\n"
        "4. Ground Truth Answer Evaluation: Is the ground truth answer correct?\n"
        "4.1. If Yes, classify as ok.\n"
        "4.2. If No, classify as 'not ok'.\n"
        "Provide your assessment in JSON format with keys 'Question Presentation', 'MC Options Presentation', 'Answer Evaluation', 'Ground Truth Answer Evaluation', 'Classification'. "
        "The 'classification' is either ok, or not ok. \n\n"
        "FOLLOW THE EXACT EXAMPLE ANSWER FORMAT WITHOUT PROVIDING EXPLANATION"
        "# Example Answer:\n"
        '{"Question Presentation": "ok", "MC Options Presentation": "ok", "Answer Evaluation": "ok", "Ground Truth Answer Evaluation": "ok", "Classification": "ok"}'
    )

    def create_conversation(example):
        messages = [
            {"role": "system", "content": DEFAULT_INSTRUCTION},
            {"role": "user", "content": example["input"] + "\nYour response:"},
            {"role": "assistant", "content": example["output"]},
        ]
        return {"messages": messages}

    CHOICES_DELIMITER = "\n"
    QUESTION_VERBALISER = "{question}\n{choices}\nAnswer: {answer}"

    def verbaliser(question, choices, answer):
        verbalised_choices = CHOICES_DELIMITER.join(
            [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)]
        )
        return QUESTION_VERBALISER.format(
            question=question,
            choices=verbalised_choices,
            answer=f"{chr(65+answer)}. {choices[answer]}",
        )

    subset_lst = [
        "bad_options_clarity",
        "bad_questions_clarity",
        "clean",
        "multiple_correct_answers",
        "no_correct_answer",
        "wrong_groundtruth",
    ]
    subset_id_to_ds = {k: load_dataset(ds_id, k) for k in subset_lst}

    remapping = {
        # "clean": "clean",
        # "bad_questions_clarity": "bad presentation",
        # "bad_options_clarity": "bad presentation",
        # "no_correct_answer": "wrong groundtruth",
        # "multiple_correct_answers": "wrong groundtruth",
        # "wrong_groundtruth": "wrong groundtruth",
        "clean": "ok",
        "bad_questions_clarity": "not ok",
        "bad_options_clarity": "not ok",
        "no_correct_answer": "not ok",
        "multiple_correct_answers": "not ok",
        "wrong_groundtruth": "not ok",
    }

    probs = [0.1, 0.1, 0.5, 0.1, 0.1, 0.1]
    # probs = [0.009, 0.031, 0.848, 0.029, 0.013, 0.07]

    for subset_id, ds in subset_id_to_ds.items():
        for split_name in ["train", "validation", "test"]:
            input_lst, output_lst = [], []

            for entry in ds[split_name]:
                input_str = verbaliser(
                    entry["question"], entry["choices"], entry["answer"]
                )
                input_lst += [input_str]
                # output_lst += [(subset_id.replace("_", " "))]
                output_lst += [remapping[subset_id]]

            ds[split_name] = ds[split_name].add_column("input", input_lst)
            ds[split_name] = ds[split_name].add_column("output", output_lst)

    # ok_train_ds = subset_id_to_ds['clean']['train']

    # not_ok_lst = [entry for entry in subset_lst if 'clean' not in entry]
    # not_ok_train_ds = concatenate_datasets([subset_id_to_ds[name]['train'] for name in not_ok_lst])

    train_ds = interleave_datasets(
        [subset_id_to_ds[name]["train"] for name in subset_lst],
        probabilities=probs,
        seed=42,
    )

    train_ds = train_ds.map(
        create_conversation,
        remove_columns=train_ds.features,
        batched=False,
        desc="Generating Yes conversations",
    )

    # not_ok_train_ds = not_ok_train_ds.map(create_conversation,
    #                                      remove_columns=not_ok_train_ds.features,
    #                                      batched=False,
    #                                      desc="Generating No conversations")

    # train_ds = concatenate_datasets([ok_train_ds, not_ok_train_ds])

    # ok_dev_ds = subset_id_to_ds['clean']['validation']
    # not_ok_dev_ds = concatenate_datasets([subset_id_to_ds[name]['validation'] for name in not_ok_lst])

    # ok_dev_ds = ok_dev_ds.map(create_conversation,
    #                          remove_columns=ok_dev_ds.features,
    #                          batched=False,
    #                          desc="Generating Yes conversations")
    #
    # not_ok_dev_ds = not_ok_dev_ds.map(create_conversation,
    #                                  remove_columns=not_ok_dev_ds.features,
    #                                  batched=False,
    #                                  desc="Generating No conversations")

    # dev_ds = interleave_datasets([ok_dev_ds, not_ok_dev_ds], probabilities=[0.5, 0.5], seed=42)
    ##dev_ds = concatenate_datasets([ok_dev_ds, not_ok_dev_ds])

    probs = [0.1, 0.1, 0.5, 0.1, 0.1, 0.1]

    dev_ds = interleave_datasets(
        [subset_id_to_ds[name]["validation"] for name in subset_lst],
        probabilities=probs,
        seed=42,
    )

    dev_ds = dev_ds.map(
        create_conversation,
        remove_columns=dev_ds.features,
        batched=False,
        desc="Generating Yes conversations",
    )

    collator = None
    if args.collator in {"completion"}:
        # _, user_template, assistant_template = extract_templates(tokenizer)
        # assert assistant_template == "<|im_start|>assistant"
        # assert user_template == "<|im_start|>user"
        if "gemma" in args.model:
            user_template = "<start_of_turn>user"
            assistant_template = "<start_of_turn>model"
        elif "Llama-3" in args.model:
            user_template = "<|start_header_id|>user<|end_header_id|>"
            assistant_template = "<|start_header_id|>assistant<|end_header_id|>"
        elif "Llama-2" in args.model:
            user_template = "<</SYS>>"
            assistant_template = "[/INST]"
        elif "Mistral" in args.model:
            user_template = "[INST]"
            assistant_template = "[/INST]"

        collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            response_template=assistant_template,
            instruction_template=user_template,
            mlm=False,
        )

    lora_config = LoraConfig(
        r=args.lora_rank,
        # target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    # lora_config = None

    # output_dir = os.path.join(args.output_dir, create_model_path(args))
    output_dir = os.path.join(args.output_dir, args.model, "unaligned")
    run_name = " ".join(sys.argv)

    logger.info(f"Output dir: {output_dir}")

    # breakpoint()

    # icd10_diag_set, icd10_proc_set = get_all_codes()

    # from minimt.evaluation import compute_metrics as compute_metrics_
    compute_metrics = None  # lambda x: compute_metrics_(x, tokenizer, model, collator)

    training_kwargs = {"bf16": True} if use_bf16 else {}
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        report_to="wandb",
        run_name=run_name,
        push_to_hub=True,
        # Training strategy
        do_train=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        optim=args.optim,
        load_best_model_at_end=True,
        save_total_limit=2,
        # Logging strategy
        logging_strategy="steps",
        logging_steps=1,
        # Evaluation strategy
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        per_device_eval_batch_size=args.batch_size,
        include_inputs_for_metrics=True,
        # Saving strategy
        save_steps=args.eval_steps,
        **training_kwargs,
    )

    # from minimt.coding.callbacks import EvaluationCallback
    # callback = EvaluationCallback(prompt="Hello world")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics,
        data_collator=collator,
        packing=False,
        # formatting_func=formatting_func,
        max_seq_length=args.max_seq_length,
        peft_config=lora_config,
        # callbacks=[callback],
        dataset_kwargs={"add_special_tokens": False},
        args=training_args,
    )

    train_result = trainer.train()

    # {'train_runtime': 241.7539, 'train_samples_per_second': 0.265, 'train_steps_per_second': 0.033, 'train_loss': 3.466222196817398, 'epoch': 0.01}
    train_metrics = train_result.metrics

    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    # {'eval_loss': 3.608168125152588, 'eval_runtime': 38.7515, 'eval_samples_per_second': 2.4, 'eval_steps_per_second': 2.4, 'epoch': 0.01}
    eval_metrics = trainer.evaluate()

    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    trainer.save_state()

    if wandb.run is not None:
        wandb.finish()

    # repo_id = 'MiniML/mini-llm_v1'
    # best_results = get_best_results(repo_id)
    #
    # if best_results is None or best_results['eval_loss'] > eval_metrics['eval_loss']:
    #    api = HfApi()

    #    results = {**train_metrics, **eval_metrics, 'command': ' '.join(sys.argv)}
    #    with open('new_best_results.json', 'w') as f:
    #        json.dump(results, f, indent=4)
    #
    #    api.upload_file(path_or_fileobj='new_best_results.json', path_in_repo='results.json', repo_id=repo_id, repo_type='model')

    #    trainer.tokenizer.push_to_hub(repo_id=repo_id, private=True)
    #    trainer.model.push_to_hub(repo_id=repo_id, private=True)


if __name__ == "__main__":
    main(sys.argv[1:])
