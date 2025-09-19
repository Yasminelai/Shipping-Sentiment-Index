#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Domain-Adaptive Pre-Training (DAPT) for RoBERTa on maritime corpus (MLM).
- Reads ../Data/maritime_corpus.csv
- Uses only the `sentence` column (other columns are ignored)
- Assumes sentences are already cleaned; only control max token length
- Saves model/tokenizer to ../Model/dapt
"""

import os
import argparse
import math
from dataclasses import dataclass
from typing import Dict, Any, List

import pandas as pd
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    set_seed,
    EarlyStoppingCallback,
)

# ------------------------------
# Utils
# ------------------------------

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DAPT (MLM) training for RoBERTa on maritime corpus")

    # Paths
    parser.add_argument("--data_path", type=str, default="../Data/maritime_corpus.csv",
                        help="Path to maritime_corpus.csv")
    parser.add_argument("--text_column", type=str, default="sentence",
                        help="Name of the text column to use")
    parser.add_argument("--output_dir", type=str, default="../Model/dapt",
                        help="Where to save the trained model/tokenizer/checkpoints")

    # Model & tokenization
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base",
                        help="HF model checkpoint to start from")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Max tokens per example (truncate longer ones)")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Masking probability for MLM")

    # Training
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_val_split", type=float, default=0.95,
                        help="Proportion of data used for training (rest for validation)")

    # Performance & precision
    parser.add_argument("--fp16", action="store_true", help="Use FP16 if available")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 if available")
    parser.add_argument("--num_proc", type=int, default=1, help="Parallelism for datasets.map")

    # Logging
    parser.add_argument("--report_to", type=str, default="none",
                        help="Reporting integration: 'none', 'wandb', 'tensorboard', etc.")
    parser.add_argument("--run_name", type=str, default="dapt_roberta_maritime",
                        help="Experiment/run name for trackers")

    # Early stopping
    parser.add_argument("--early_stopping", action="store_true",
                        help="Enable early stopping on validation loss")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="Number of evaluations with no improvement before stopping")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0,
                        help="Minimum improvement in monitored metric to reset patience")

    # Checkpoint / resume
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a specific checkpoint to resume from (overrides auto-detect)")
    parser.add_argument("--auto_resume", action="store_true",
                        help="Auto-detect latest checkpoint under output_dir/checkpoint-* to resume")

    return parser.parse_args()


def build_dataset(args: argparse.Namespace, tokenizer) -> DatasetDict:
    """
    Load CSV via HF datasets, keep only the text column, drop empties/NaNs,
    and return tokenized train/validation splits.
    """
    # Load as a single "train" split
    ds = load_dataset("csv", data_files=args.data_path)["train"]

    # Keep only the text column (in case CSV has multiple columns)
    # We do not error if the column exists alongside others; we just reference it.
    if args.text_column not in ds.column_names:
        raise ValueError(f"Column '{args.text_column}' not found in: {ds.column_names}")

    # Basic cleaning filter: drop empty/NaN rows for the target column
    def _valid_example(example):
        txt = example.get(args.text_column, None)
        return isinstance(txt, str) and len(txt.strip()) > 0

    ds = ds.filter(_valid_example)

    # Train/validation split
    split = ds.train_test_split(test_size=1 - args.train_val_split, seed=args.seed)
    raw_train = split["train"]
    raw_eval = split["test"]

    # Tokenization function
    def tok(batch):
        # Only use the target column; any other columns are ignored
        return tokenizer(
            batch[args.text_column],
            truncation=True,
            max_length=args.max_length,
            padding=False,  # dynamic padding via data collator
        )

    # Map with batching & optional multiprocessing
    tokenized_train = raw_train.map(tok, batched=True, num_proc=args.num_proc, remove_columns=raw_train.column_names)
    tokenized_eval = raw_eval.map(tok, batched=True, num_proc=args.num_proc, remove_columns=raw_eval.column_names)

    return DatasetDict(train=tokenized_train, validation=tokenized_eval)


def compute_perplexity(eval_loss: float) -> float:
    """Convert eval loss (cross-entropy) to perplexity; guard against overflow."""
    try:
        return float(math.exp(eval_loss))
    except OverflowError:
        return float("inf")


def main():
    args = get_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)

    # Build dataset (only uses `sentence` column by default)
    dsdict = build_dataset(args, tokenizer)

    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    # Mixed precision flags
    fp16_flag = bool(args.fp16 and torch.cuda.is_available())
    bf16_flag = bool(args.bf16 and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8)

    # Training arguments
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

        eval_strategy="steps",
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

    # Trainer
    callbacks = []
    if args.early_stopping:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        ))

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dsdict["train"],
        eval_dataset=dsdict["validation"],
        tokenizer=tokenizer,
        callbacks=callbacks if callbacks else None,
    )

    # Initial evaluation (Step 0 baseline)
    print(">>> Evaluating before training...")
    eval_metrics = trainer.evaluate()
    base_loss = eval_metrics.get("eval_loss", None)
    if base_loss is not None:
        print(f"Initial eval loss: {base_loss:.4f} | ppl: {compute_perplexity(base_loss):.2f}")

    # Train with resume support
    print(">>> Training...")
    resume_path = None
    if args.resume_from_checkpoint:
        resume_path = args.resume_from_checkpoint
    elif args.auto_resume:
        # Auto-detect latest checkpoint in output_dir
        try:
            candidates = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
            if candidates:
                latest = sorted(candidates, key=lambda x: int(x.split("-")[-1]))[-1]
                resume_path = os.path.join(args.output_dir, latest)
        except Exception:
            resume_path = None

    if resume_path and os.path.isdir(resume_path):
        print(f">>> Resuming from checkpoint: {resume_path}")
        trainer.train(resume_from_checkpoint=resume_path)
    else:
        trainer.train()

    # Final evaluation
    print(">>> Evaluating after training...")
    eval_metrics = trainer.evaluate()
    final_loss = eval_metrics.get("eval_loss", None)
    if final_loss is not None:
        print(f"Final eval loss: {final_loss:.4f} | ppl: {compute_perplexity(final_loss):.2f}")

    # Save model & tokenizer
    print(f">>> Saving model & tokenizer to: {args.output_dir}")
    trainer.save_model(args.output_dir)        # saves model + config
    tokenizer.save_pretrained(args.output_dir) # saves tokenizer files

    print(">>> Done.")


if __name__ == "__main__":
    main()
