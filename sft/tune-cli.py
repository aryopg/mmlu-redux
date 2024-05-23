#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import torch
import argparse
import logging

import wandb

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, EvalPrediction
from transformers.utils import is_bitsandbytes_available, is_flash_attn_2_available

from datasets import load_dataset, interleave_datasets

from trl import SFTTrainer, setup_chat_format, DataCollatorForCompletionOnlyLM
from peft import LoraConfig

from huggingface_hub import HfApi

from minimt.utils import check_bf16_support, extract_templates, create_model_path, get_best_results
from minimt.coding.utils import get_all_codes

from typing import Dict

os.environ['TOKENIZERS_PARALLELISM'] = "false"
os.environ["WANDB_PROJECT"] = 'mini-llm_v1'

logger = logging.getLogger(name=__file__)
logger.setLevel(logging.INFO)

# For debugging/prototyping:
# PYTHONPATH=. ./cli/mt-sft-cli.py --use-bf16 --learning-rate 1e-4 --batch-size 4 --collator completion --gradient-accumulation-steps 1 --max-steps 8 --model google/gemma-2b --lora-rank 8 --warmup-steps 4 --max-seq-length 4096

def main(argv):
    parser = argparse.ArgumentParser(description="Note-to-FHIR Trainer Model")

    parser.add_argument("--model", "-m", type=str, default="mistralai/Mistral-7B-v0.1", help="Base model ID")

    parser.add_argument("--fhir-dataset", type=str, default="healthsageai/fhir-to-note", help="FHIR dataset")
    parser.add_argument("--icd10-dataset", type=str, default="MiniML/mimiciv-icd10", help="ICD10 dataset")

    parser.add_argument("--output-dir", "-o", type=str, default="outputs", help="Output directory for training artifacts")
    parser.add_argument("--use-bf16", "--bf16", action="store_true", help="BF16 support if available")
    parser.add_argument("--use-quantization", "-q", action="store_true", help="Enable quantization with BitsAndBytes")

    parser.add_argument("--optim", type=str, default="adamw_torch", help="Optimizer")
    parser.add_argument("--eval-steps", type=int, default=8, help="Evaluation steps")
    parser.add_argument("--learning-rate", "--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Batch size")
    parser.add_argument("--gradient-accumulation-steps", "-g", type=int, default=8, help="No. of gradient accumulation steps")
    parser.add_argument("--warmup-steps", "-w", type=int, default=32, help="No. of warmup steps")
    
    parser.add_argument("--max-steps", type=int, default=1024, help="Max no. of training steps")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Max sequence length")

    parser.add_argument("--collator", action="store", type=str, default="completion", choices=["completion", "default"])
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")

    # Parse arguments
    args = parser.parse_args()

    use_bf16 = check_bf16_support(args.use_bf16)

    model_id = args.model

    bnb_config = None
    if args.use_quantization and is_bitsandbytes_available():
        logger.info("Using quantisation ..")
        bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                        bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16)
        model_kwargs = {'quantization_config': bnb_config}
    else:
        model_kwargs = {'torch_dtype': torch.bfloat16} if use_bf16 else {}

    if is_flash_attn_2_available() and use_bf16:
        logger.info("Using Flash Attention ..")
        model_kwargs.update({"attn_implementation": "flash_attention_2"})

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 device_map="auto",
                                                 low_cpu_mem_usage=True,
                                                 trust_remote_code=True,
                                                 use_cache=False,
                                                 **model_kwargs)

    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")

    fhir_ds = load_dataset(args.fhir_dataset)
    icd10_ds = load_dataset(args.icd10_dataset)

    DEFAULT_INSTRUCTION_FHIR = "Convert the clinical note below into HL7 FHIR R4 format in JSON, adhering to these guidelines. Include only data explicitly mentioned in the note. Avoid adding, inferring, or imputing values not present in the text. Ensure to incorporate all information provided in the clinical note and mandatory fields required for a valid FHIR resource. Your output must strictly conform to the FHIR R4 JSON structure, focusing on accuracy and adherence to the specified FHIR standards."

    DEFAULT_INSTRUCTION_ICD10 = "Extract the ICD-10 codes from the following clinical text. Use the information provided in the text and apply relevant medical knowledge to identify the appropriate ICD-10 codes. The extracted codes should be specific to the conditions mentioned in the text and adhere to the ICD-10 classification system. Avoid assumptions or inferences not supported by the text. Ensure that the codes reflect the diagnoses, symptoms, or reasons for visit as described. Provide the codes as a list in the response."

    def create_conversation_fhir(example):
        messages = [
            {"role": "system", "content": DEFAULT_INSTRUCTION_FHIR},
            {"role": "user", "content": example["note"]},
            {"role": "assistant", "content": example["fhir"]}
        ]
        return {"messages": messages}
    
    def create_conversation_icd10(example):
        target = ', '.join(example["target"])
        messages = [
            {"role": "system", "content": DEFAULT_INSTRUCTION_ICD10},
            {"role": "user", "content": example["text"]},
            {"role": "assistant", "content": target}
        ]
        return {"messages": messages}

    fhir_ds = fhir_ds.map(create_conversation_fhir,
                          remove_columns=fhir_ds["train"].features,
                          batched=False,
                          desc="Generating FHIR conversations")
    
    icd10_ds = icd10_ds.map(create_conversation_icd10,
                            remove_columns=icd10_ds["train"].features,
                            batched=False,
                            desc="Generating ICD10 conversations")

    train_ds = interleave_datasets([fhir_ds["train"], icd10_ds["train"]], probabilities=[0.5, 0.5], seed=42)
    val_ds = interleave_datasets([fhir_ds["validation"], icd10_ds["val"]], probabilities=[0.5, 0.5], seed=42)

    collator = None
    if args.collator in {"completion"}:
        _, user_template, assistant_template = extract_templates(tokenizer)
        assert assistant_template == "<|im_start|>assistant"
        assert user_template == "<|im_start|>user"

        collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer,
                                                   response_template=assistant_template,
                                                   instruction_template=user_template,
                                                   mlm=False)

    lora_config = LoraConfig(r=args.lora_rank,
                             # target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
                             bias="none",
                             task_type="CAUSAL_LM")

    output_dir = os.path.join(args.output_dir, create_model_path(args))
    run_name = ' '.join(sys.argv)

    logger.info(f'Output dir: {output_dir}')

    # breakpoint()

    icd10_diag_set, icd10_proc_set = get_all_codes()

    from minimt.evaluation import compute_metrics as compute_metrics_
    compute_metrics = None # lambda x: compute_metrics_(x, tokenizer, model, collator)

    training_kwargs = {'bf16': True} if use_bf16 else {}
    training_args = TrainingArguments(output_dir=output_dir,
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
                                      **training_kwargs)

    # from minimt.coding.callbacks import EvaluationCallback
    # callback = EvaluationCallback(prompt="Hello world")

    trainer = SFTTrainer(model=model,
                         tokenizer=tokenizer,
                         train_dataset=train_ds,
                         eval_dataset=val_ds,
                         compute_metrics=compute_metrics,
                         data_collator=collator,
                         packing=False,
                         max_seq_length=args.max_seq_length,
                         peft_config=lora_config,
                         # callbacks=[callback],
                         dataset_kwargs={'add_special_tokens': False},
                         args=training_args)

    train_result = trainer.train()

    # {'train_runtime': 241.7539, 'train_samples_per_second': 0.265, 'train_steps_per_second': 0.033, 'train_loss': 3.466222196817398, 'epoch': 0.01}
    train_metrics = train_result.metrics
        
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)

    # {'eval_loss': 3.608168125152588, 'eval_runtime': 38.7515, 'eval_samples_per_second': 2.4, 'eval_steps_per_second': 2.4, 'epoch': 0.01}
    eval_metrics = trainer.evaluate()

    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    trainer.save_state()

    if wandb.run is not None:
        wandb.finish()

    repo_id = 'MiniML/mini-llm_v1'
    best_results = get_best_results(repo_id)
    
    if best_results is None or best_results['eval_loss'] > eval_metrics['eval_loss']:
        api = HfApi()

        results = {**train_metrics, **eval_metrics, 'command': ' '.join(sys.argv)}
        with open('new_best_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        api.upload_file(path_or_fileobj='new_best_results.json', path_in_repo='results.json', repo_id=repo_id, repo_type='model')

        trainer.tokenizer.push_to_hub(repo_id=repo_id, private=True)
        trainer.model.push_to_hub(repo_id=repo_id, private=True)


if __name__ == '__main__':
    main(sys.argv[1:])
