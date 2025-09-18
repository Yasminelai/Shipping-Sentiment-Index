"""
Task-Adaptive Pre-Training (TAPT) for RoBERTa on news corpus (MLM).
- Reads ../Data/News_corpus.csv
- Uses only the `text` column (other columns such as date, url, title are ignored)
- Assumes text is already cleaned; we only control max token length
- Initializes from the DAPT checkpoint at ../Model/dapt_model (continued pretraining)
- Saves model/tokenizer to ../Model/tapt_model
"""

import os
import argparse
import math
from typing import Dict
from datasets import load_dataset, DatasetDict
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    set_seed,
)

def get_args():
    p = argparse.ArgumentParser("TAPT (MLM) continuing from DAPT checkpoint")
    # Paths
    p.add_argument("--data_path", type=str, default="../Data/News_corpus.csv")
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--init_from", type=str, default="../Model/dapt_model",  # ← DAPT output
                   help="Initialize TAPT from this DAPT checkpoint dir")
    p.add_argument("--output_dir", type=str, default="../Model/tapt_model")
    # Tokenization / MLM
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--mlm_probability", type=float, default=0.15)
    # Training
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--per_device_eval_batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--eval_steps", type=int, default=1000)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_val_split", type=float, default=0.95)
    # Precision
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--num_proc", type=int, default=1)
    # Logging
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--run_name", type=str, default="tapt_from_dapt_news")
    return p.parse_args()

def build_dataset(args, tokenizer) -> DatasetDict:
    ds = load_dataset("csv", data_files=args.data_path)["train"]

    if args.text_column not in ds.column_names:
        raise ValueError(f"Column '{args.text_column}' not found in: {ds.column_names}")

    # keep non-empty rows
    ds = ds.filter(lambda ex: isinstance(ex.get(args.text_column, ""), str) and len(ex[args.text_column].strip()) > 0)

    split = ds.train_test_split(test_size=1 - args.train_val_split, seed=args.seed)
    raw_train, raw_eval = split["train"], split["test"]

    def tok(batch):
        return tokenizer(
            batch[args.text_column],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    tokenized_train = raw_train.map(tok, batched=True, num_proc=args.num_proc, remove_columns=raw_train.column_names)
    tokenized_eval  = raw_eval.map(tok,  batched=True, num_proc=args.num_proc, remove_columns=raw_eval.column_names)

    return DatasetDict(train=tokenized_train, validation=tokenized_eval)

def perplexity(loss: float) -> float:
    try:
        import math
        return float(math.exp(loss))
    except OverflowError:
        return float("inf")

def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize tokenizer & model FROM DAPT checkpoint
    tokenizer = AutoTokenizer.from_pretrained(args.init_from, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(args.init_from)

    dsdict = build_dataset(args, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability
    )

    fp16_flag = bool(args.fp16 and torch.cuda.is_available())
    bf16_flag = bool(args.bf16 and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        fp16=fp16_flag,
        bf16=bf16_flag,
        dataloader_num_workers=2,
        report_to=None if args.report_to == "none" else args.report_to,
        run_name=args.run_name,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dsdict["train"],
        eval_dataset=dsdict["validation"],
        tokenizer=tokenizer,
    )

    print(">>> Evaluating before training...")
    base = trainer.evaluate()
    if "eval_loss" in base:
        print(f"Initial loss: {base['eval_loss']:.4f} | ppl: {perplexity(base['eval_loss']):.2f}")

    print(">>> Training (TAPT from DAPT)...")
    trainer.train()

    print(">>> Evaluating after training...")
    final = trainer.evaluate()
    if "eval_loss" in final:
        print(f"Final loss: {final['eval_loss']:.4f} | ppl: {perplexity(final['eval_loss']):.2f}")

    print(f">>> Saving TAPT model & tokenizer to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(">>> Done.")

if __name__ == "__main__":
    main()
