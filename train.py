import argparse
import re
from functools import partial

import torch
import yaml
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


def tokenize_normal(example, tokenizer):
    messages = example["messages"]
    tokenized = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=True)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def tokenizer_ignore_user_messages(example, tokenizer):
    messages = example["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    tokenized = tokenizer(text, add_special_tokens=False)

    pattern = re.escape("<|assistant|>\n") + r"(.*?" + re.escape(tokenizer.eos_token) + ")"
    assistent_start_end = [(m.start(1), m.end(1)) for m in re.finditer(pattern, text, re.DOTALL)]

    labels = [-100] * len(tokenized["input_ids"])
    for start, end in assistent_start_end:
        for i in range(start, end):
            token = tokenized.char_to_token(i)
            labels[token] = tokenized["input_ids"][token]
    tokenized["labels"] = labels
    return tokenized


def train(config: dict, ignore_user_messages: bool):

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.chat_template = config["chat_template"]
    if ignore_user_messages:
        tokenizer_fn = partial(tokenizer_ignore_user_messages, tokenizer=tokenizer)
    else:
        tokenizer_fn = partial(tokenize_normal, tokenizer=tokenizer)

    dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
    dataset.pop("test_gen")
    dataset.pop("train_gen")
    dataset = dataset.map(tokenizer_fn, batched=False, num_proc=12, remove_columns=["prompt", "prompt_id", "messages"])

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=False,
        quantization_config=quantization_config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", help="Path to config.yaml", required=True)
    parser.add_argument("--ignore-user-messages", help="Ignore user tokens in training", required=True, type=bool)

    args, unknown_args = parser.parse_known_args()
    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    train(config, args.ignore_user_messages)
