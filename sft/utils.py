# -*- coding: utf-8 -*-

import re
import hashlib

import json
import torch
from torch.utils.data import Dataset

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
import transformers
from transformers import AutoTokenizer

from typing import Tuple, Dict


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

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

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

def extract_templates(tokenizer) -> Tuple[str, str, str]:
    system_prompt, user_prompt, assistant_prompt = map(checksum, [f"{e} prompt" for e in ["system", "user", "assistant"]])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    def extract_template(prompt: str) -> str:
        expression = rf'\n(.*?)\s*{prompt}'
        match = re.search(expression, text)
        assert match is not None
        res = match.group(1)
        return f"{res}"
    
    res = map(extract_template, [system_prompt, user_prompt, assistant_prompt])
    return tuple(res)

def create_model_path(args):
    # Sanitize the model ID to replace '/' with '_' to avoid directory path issues
    sanitized_model_id = re.sub(r'[/\\]', '_', args.model.split("/")[-1])
    
    path_parts = [
        sanitized_model_id,
        f"bs={args.batch_size}",
        f"lr={args.learning_rate:.0e}",
        f"collator={args.collator}",
        f"gas={args.gradient_accumulation_steps}",
        f"ms={args.max_steps}",
        f"lrank={args.lora_rank}",
        f"ws={args.warmup_steps}"
    ]
    
    # Combine all parts with underscores
    path = "_".join(path_parts)

    # Sanitise, due to the following:
    # huggingface_hub.utils._validators.HFValidationError: Repo id must [..]
    # path = path.replace('_', '-')
    path = path.replace('=', '_')
    
    return path


def get_best_results(repo_id: str) -> Dict[str, float]:
    best_results = None
    try:
        best_results_path = hf_hub_download(repo_id=repo_id, filename="results.json", repo_type="model")
        
        if best_results_path is not None:
            with open(best_results_path, 'r') as f:
                best_results = json.load(f)
    except (EntryNotFoundError, RepositoryNotFoundError):
        pass
    return best_results
