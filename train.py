import argparse
import re
from functools import partial

import torch
import yaml
from accelerate.utils import calculate_maximum_sizes
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def setup_chat_format(model, tokenizer, config):
    tokenizer.chat_template = config["chat_template"]
    tokenizer.add_special_tokens({"additional_special_tokens": config["tokenizer_special_tokens"]})
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def tokenize_normal(example, tokenizer):
    messages = example["messages"]
    tokenized = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_dict=True, truncation=True
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def tokenize_ignore_user_messages(example, tokenizer):
    # strong assumption on chatml format where the assistant messages are between <|assistant|> and <|im_end|>
    messages = example["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    tokenized = tokenizer(text, add_special_tokens=False, truncation=True)

    pattern = re.escape("<|im_start|>assistant\n") + r"(.*?" + re.escape("<|im_end|>") + ")"
    assistent_start_end = [(m.start(1), m.end(1)) for m in re.finditer(pattern, text, re.DOTALL)]

    labels = [-100] * len(tokenized["input_ids"])
    for start, end in assistent_start_end:
        start_token = tokenized.char_to_token(start)
        end_token = tokenized.char_to_token(end - 1)
        if start_token is None:
            break  # start is after truncated text
        for token_id in range(start_token, end_token + 1 if end_token else len(tokenized["input_ids"])):
            labels[token_id] = tokenized["input_ids"][token_id]
    tokenized["labels"] = labels
    return tokenized


def fix_universal_ner(example):
    return {
        "messages": [
            {"content": m["value"], "role": "assistant" if m["from"] == "gpt" else "user"}
            for m in example["conversations"]
        ]
    }


def datasets(config, tokenizer, ignore_user_messages: bool):

    train_tokenizer_fn = tokenize_ignore_user_messages if ignore_user_messages else tokenize_normal

    dataset = load_dataset(config["dataset"]["name"])

    if config["dataset"]["name"] == "Universal-NER/Pile-NER-type":
        dataset = dataset.map(fix_universal_ner, batched=False, num_proc=12, remove_columns=["conversations"])["train"]
        dataset = dataset.train_test_split(test_size=0.02, seed=42)

    columns = ["input_ids", "labels", "attention_mask"]
    train_ds = dataset[config["dataset"]["train_split"]].map(
        train_tokenizer_fn,
        batched=False,
        num_proc=12,
        remove_columns=[i for i in dataset[config["dataset"]["train_split"]].column_names if i not in columns],
        fn_kwargs={"tokenizer": tokenizer},
        load_from_cache_file=True,
    )

    # for the validation I want to track the loss of the assistant messages only!
    val_ds = dataset[config["dataset"]["val_split"]].map(
        tokenize_ignore_user_messages,
        batched=False,
        num_proc=12,
        remove_columns=[i for i in dataset[config["dataset"]["val_split"]].column_names if i not in columns],
        fn_kwargs={"tokenizer": tokenizer},
        load_from_cache_file=True,
    )

    return train_ds, val_ds


def get_model(config):
    quantization_config = BitsAndBytesConfig(**config["quantization"])
    config["model_args"]["torch_dtype"] = getattr(torch, config["model_args"]["torch_dtype"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        quantization_config=quantization_config,
        **config["model_args"],
    )

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(**config["lora"])

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("model size GB ", round(calculate_maximum_sizes(model)[0] / 1024 / 1024 / 1024, 2))

    return model


def collate_fn(examples, pad_token_id):

    max_length = max(len(e["input_ids"]) for e in examples)
    padded_examples = []
    for example in examples:
        input_ids = example["input_ids"]
        labels = example["labels"]
        attention_mask = example["attention_mask"]

        difference = max_length - len(example["labels"])
        if difference != 0:
            # right padding. consider left?
            input_ids = input_ids + difference * [pad_token_id]
            labels = labels + difference * [-100]
            attention_mask = attention_mask + difference * [0]
        padded_examples.append({"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask})

    batch = {
        "input_ids": torch.tensor([e["input_ids"] for e in padded_examples]),
        "labels": torch.tensor([e["labels"] for e in padded_examples]),
        "attention_mask": torch.tensor([e["attention_mask"] for e in padded_examples]),
    }

    return batch


def train(config: dict, ignore_user_messages: bool):
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    tokenizer.model_max_length = 2048
    assert tokenizer.chat_template is None, "modify `tokenizer_ignore_user_messages` to handle specific chat template."

    model = get_model(config)
    model, tokenizer = setup_chat_format(model, tokenizer, config)

    train_ds, val_ds = datasets(config, tokenizer, ignore_user_messages)

    args = TrainingArguments(**config["training_args"])
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config yaml path")
    parser.add_argument(
        "--ignore-user-messages",
        help="mask user tokens",
        required=True,
        type=bool,
        action=argparse.BooleanOptionalAction,
    )

    args, unknown_args = parser.parse_known_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train(config, args.ignore_user_messages)
