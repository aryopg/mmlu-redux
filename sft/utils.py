# -*- coding: utf-8 -*-

import re
import hashlib

import torch
from torch.utils.data import Dataset

import transformers
from transformers import AutoTokenizer

from typing import Dict


class NestedKeyDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        key: str,
        tokenizer: AutoTokenizer,
    ):
        self.dataset = dataset
        self.key = key

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        prompt = self.tokenizer.apply_chat_template(
            self.dataset[i][self.key][:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def check_bf16_support(use_bf16_arg):
    """Check for BF16 support on the system."""
    res = False
    if torch.cuda.is_available() and use_bf16_arg:
        res = torch.cuda.is_bf16_supported()
    return res


def checksum(text: str) -> str:
    return hashlib.sha512(text.encode()).hexdigest()


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
