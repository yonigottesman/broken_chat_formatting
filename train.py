import argparse
import re
from functools import partial

import torch
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


def tokenize_normal(example, tokenizer):
    messages = example["messages"]
    tokenized = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_dict=True, truncation=True
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def tokenizer_ignore_user_messages(example, tokenizer):
    messages = example["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    tokenized = tokenizer(text, add_special_tokens=False, truncation=True)

    pattern = re.escape("<|assistant|>\n") + r"(.*?" + re.escape("<|end|>") + ")"
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


def datasets(tokenizer, ignore_user_messages: bool):

    train_tokenizer_fn = tokenizer_ignore_user_messages if ignore_user_messages else tokenize_normal

    dataset = load_dataset("HuggingFaceH4/ultrachat_200k")

    train_ds = dataset["train_sft"].map(
        train_tokenizer_fn,
        batched=False,
        num_proc=12,
        remove_columns=["prompt", "prompt_id", "messages"],
        fn_kwargs={"tokenizer": tokenizer},
        load_from_cache_file=True,
    )

    # for the validation I want to track the loss of the assistant messages only!
    val_ds = dataset["test_sft"].map(
        tokenizer_ignore_user_messages,
        batched=False,
        num_proc=12,
        remove_columns=["prompt", "prompt_id", "messages"],
        fn_kwargs={"tokenizer": tokenizer},
        load_from_cache_file=True,
    )

    return train_ds, val_ds


def get_model():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        quantization_config=quantization_config,
        use_cache=False,
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("model size GB ", calculate_maximum_sizes(model)[0] / 1024 / 1024 / 1024)

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


def train(ignore_user_messages: bool):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    model = get_model()

    train_ds, val_ds = datasets(tokenizer, ignore_user_messages)

    args = TrainingArguments(
        output_dir="./output/",
        bf16=True,
        do_eval=True,
        evaluation_strategy="epoch",
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=5.0e-06,
        logging_steps=20,
        logging_strategy="steps",
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        max_steps=-1,
        overwrite_output_dir=True,
        per_device_eval_batch_size=2,
        per_device_train_batch_size=2,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        warmup_ratio=0.2,
        report_to="none",
    )
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
    parser.add_argument("--ignore-user-messages", help="Ignore user tokens in training", required=True, type=bool)

    args, unknown_args = parser.parse_known_args()
    train(args.ignore_user_messages)
