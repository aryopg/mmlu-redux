#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import argparse
import logging

import re


from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from transformers.utils import is_bitsandbytes_available, is_flash_attn_2_available

from datasets import load_dataset, interleave_datasets, concatenate_datasets

from trl import SFTTrainer, setup_chat_format, DataCollatorForCompletionOnlyLM
from peft import LoraConfig


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_PROJECT"] = "mmlu-llm_v1"

logger = logging.getLogger(name=__file__)
logger.setLevel(logging.INFO)


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


# For debugging/prototyping:
# PYTHONPATH=. ./cli/mt-sft-cli.py --use-bf16 --learning-rate 1e-4 --batch-size 4 --collator completion --gradient-accumulation-steps 1 --max-steps 8 --model google/gemma-2b --lora-rank 8 --warmup-steps 4 --max-seq-length 4096


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
        "--fhir-dataset",
        type=str,
        default="healthsageai/fhir-to-note",
        help="FHIR dataset",
    )
    parser.add_argument(
        "--icd10-dataset",
        type=str,
        default="MiniML/mimiciv-icd10",
        help="ICD10 dataset",
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
        "--warmup-steps", "-w", type=int, default=32, help="No. of warmup steps"
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

    use_bf16 = args.use_bf16

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

    DEFAULT_INSTRUCTION = "Analyse carefully the following multiple-choice question and corresponding answer, and tell me whether it is correct."

    def create_conversation(example):
        messages = [
            {"role": "system", "content": DEFAULT_INSTRUCTION},
            {"role": "user", "content": example["input"]},
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

    for subset_id, ds in subset_id_to_ds.items():
        for split_name in ["train", "validation", "test"]:
            input_lst, output_lst = [], []

            for entry in ds[split_name]:
                input_str = verbaliser(
                    entry["question"], entry["choices"], entry["answer"]
                )
                input_lst += [input_str]
                output_lst += [("yes" if "clean" in subset_id else "no")]

            ds[split_name] = ds[split_name].add_column("input", input_lst)
            ds[split_name] = ds[split_name].add_column("output", output_lst)

    ok_train_ds = subset_id_to_ds["clean"]["train"]

    not_ok_lst = [entry for entry in subset_lst if "clean" not in entry]
    not_ok_train_ds = concatenate_datasets(
        [subset_id_to_ds[name]["train"] for name in not_ok_lst]
    )

    ok_train_ds = ok_train_ds.map(
        create_conversation,
        remove_columns=ok_train_ds.features,
        batched=False,
        desc="Generating Yes conversations",
    )

    not_ok_train_ds = not_ok_train_ds.map(
        create_conversation,
        remove_columns=not_ok_train_ds.features,
        batched=False,
        desc="Generating No conversations",
    )

    train_ds = interleave_datasets(
        [ok_train_ds, not_ok_train_ds], probabilities=[0.5, 0.5], seed=42
    )

    ok_dev_ds = subset_id_to_ds["clean"]["validation"]
    not_ok_dev_ds = concatenate_datasets(
        [subset_id_to_ds[name]["validation"] for name in not_ok_lst]
    )

    ok_dev_ds = ok_dev_ds.map(
        create_conversation,
        remove_columns=ok_dev_ds.features,
        batched=False,
        desc="Generating Yes conversations",
    )

    not_ok_dev_ds = not_ok_dev_ds.map(
        create_conversation,
        remove_columns=not_ok_dev_ds.features,
        batched=False,
        desc="Generating No conversations",
    )

    dev_ds = interleave_datasets(
        [ok_dev_ds, not_ok_dev_ds], probabilities=[0.5, 0.5], seed=42
    )

    collator = None
    if args.collator in {"completion"}:
        assistant_template = "<|im_start|>assistant"
        user_template = "<|im_start|>user"

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

    output_dir = os.path.join(args.output_dir, create_model_path(args))
    run_name = " ".join(sys.argv)

    logger.info(f"Output dir: {output_dir}")

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
        data_collator=collator,
        packing=False,
        max_seq_length=args.max_seq_length,
        peft_config=lora_config,
        # callbacks=[callback],
        dataset_kwargs={"add_special_tokens": False},
        args=training_args,
    )

    trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
