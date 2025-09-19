"""
Classifier fine-tuning for RoBERTa using the TAPT checkpoint (supervised).
- Reads ../Data/Annotaion.csv
- Uses only the `sentence` (text) and `label` columns (other columns are ignored)
- Assumes text is already cleaned; we only control max token length
- Loads the TAPT-trained model from ../Model/tapt and freezes the first 9 encoder layers by default
- Saves the fine-tuned classifier to ../Model/classifier
"""

import os
import argparse
from typing import Dict, Tuple

import numpy as np
import torch
from datasets import load_dataset, ClassLabel, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa classifier using TAPT checkpoint")
    # Paths
    parser.add_argument("--data_path", type=str, default="../Data/Annotaion.csv",
                        help="Path to the annotated CSV (expects columns: sentence, label; others ignored)")
    parser.add_argument("--text_column", type=str, default="sentence", help="Name of the text column")
    parser.add_argument("--label_column", type=str, default="label", help="Name of the label column")
    parser.add_argument("--tapt_dir", type=str, default="../Model/tapt",
                        help="Directory of the TAPT-trained checkpoint (model+tokenizer)")
    parser.add_argument("--output_dir", type=str, default="../Model/classifier",
                        help="Where to save the fine-tuned classifier")

    # Tokenization
    parser.add_argument("--max_length", type=int, default=256, help="Max tokens per example (truncate longer ones)")

    # Freezing
    parser.add_argument("--freeze_encoder_layers", type=int, default=9,
                        help="Freeze the first N encoder layers (default: 9)")

    # Training hyperparams
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["steps", "epoch"])
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["steps", "epoch"])
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_val_split", type=float, default=0.9,
                        help="Proportion of data used for training (rest for validation)")

    # Precision / perf
    parser.add_argument("--fp16", action="store_true", help="Use FP16 if available")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 if available")
    parser.add_argument("--num_proc", type=int, default=1, help="Parallelism for datasets.map")

    # Logging
    parser.add_argument("--report_to", type=str, default="none",
                        help="Reporting integration: 'none', 'wandb', 'tensorboard', etc.")
    parser.add_argument("--run_name", type=str, default="classifier_roberta_from_tapt",
                        help="Experiment/run name")

    # Steps (only used if strategy='steps')
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)

    # Early stopping
    parser.add_argument("--early_stopping", action="store_true",
                        help="Enable early stopping on validation loss")
    parser.add_argument("--early_stopping_patience", type=int, default=2,
                        help="Number of evaluations with no improvement before stopping")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0,
                        help="Minimum improvement in monitored metric to reset patience")

    return parser.parse_args()


def build_dataset(
    args: argparse.Namespace,
    tokenizer
) -> Tuple[DatasetDict, Dict[str, int], Dict[int, str]]:
    """
    Load CSV via HF datasets, keep only text/label columns (others ignored),
    infer label set, and return tokenized train/validation splits with proper label mapping.
    """
    ds = load_dataset("csv", data_files=args.data_path)["train"]

    # Sanity checks
    for col in (args.text_column, args.label_column):
        if col not in ds.column_names:
            raise ValueError(f"Column '{col}' not found in: {ds.column_names}")

    # Filter out invalid rows
    def _valid(example):
        txt = example.get(args.text_column, "")
        lbl = example.get(args.label_column, None)
        return isinstance(txt, str) and len(txt.strip()) > 0 and lbl is not None

    ds = ds.filter(_valid)

    # Build ClassLabel
    labels_list = sorted(list(set(ds[args.label_column])))
    if isinstance(labels_list[0], str):
        class_label = ClassLabel(names=labels_list)
    else:
        uniq = sorted(labels_list)
        class_label = ClassLabel(num_classes=len(uniq), names=[str(u) for u in uniq])

    id2label = {i: name for i, name in enumerate(class_label.names)}
    label2id = {name: i for i, name in enumerate(class_label.names)}

    # Normalize labels to ids
    def to_label_id(example):
        raw = example[args.label_column]
        if isinstance(raw, str):
            example[args.label_column] = label2id[raw]
        else:
            example[args.label_column] = label2id[str(raw)]
        return example

    ds = ds.map(to_label_id)

    # Cast label column to ClassLabel so that stratified split is supported
    try:
        ds = ds.cast_column(args.label_column, class_label)
    except Exception as e:
        print(f"Warning: failed to cast '{args.label_column}' to ClassLabel; proceeding without cast: {e}")

    # Train/validation split
    split = ds.train_test_split(
        test_size=1 - args.train_val_split,
        seed=args.seed,
        stratify_by_column=args.label_column
    )
    train_raw, eval_raw = split["train"], split["test"]

    # Tokenization
    def tok(batch):
        return tokenizer(
            batch[args.text_column],
            truncation=True,
            max_length=args.max_length,
            padding=False,  # dynamic padding via data collator in Trainer
        )

    remove_cols = [c for c in train_raw.column_names if c not in (args.text_column, args.label_column)]
    tokenized_train = train_raw.map(tok, batched=True, num_proc=args.num_proc, remove_columns=remove_cols)
    tokenized_eval = eval_raw.map(tok, batched=True, num_proc=args.num_proc, remove_columns=remove_cols)

    # Rename label column for Trainer
    tokenized_train = tokenized_train.rename_column(args.label_column, "labels")
    tokenized_eval = tokenized_eval.rename_column(args.label_column, "labels")

    return DatasetDict(train=tokenized_train, validation=tokenized_eval), label2id, id2label


def freeze_first_n_layers(model, n_layers: int) -> None:
    """
    Freeze embeddings and the first n encoder layers for RoBERTa-like architectures.
    """
    if hasattr(model, "roberta"):
        base = model.roberta
    elif hasattr(model, "bert"):
        base = model.bert
    else:
        base = getattr(model, "base_model", None)

    if base is None:
        print("Warning: could not locate base encoder to freeze; skipping.")
        return

    if hasattr(base, "embeddings"):
        for p in base.embeddings.parameters():
            p.requires_grad = False

    encoder = getattr(base, "encoder", None)
    if encoder is not None and hasattr(encoder, "layer"):
        for i, layer in enumerate(encoder.layer):
            if i < n_layers:
                for p in layer.parameters():
                    p.requires_grad = False


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_micro": f1_score(labels, preds, average="micro"),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }


def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer from TAPT (falls back to model name if needed)
    tokenizer = AutoTokenizer.from_pretrained(args.tapt_dir, use_fast=True)

    # Build dataset and label mappings
    dsdict, label2id, id2label = build_dataset(args, tokenizer)
    num_labels = len(id2label)

    # Load model from TAPT and attach a classification head
    model = AutoModelForSequenceClassification.from_pretrained(
        args.tapt_dir,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # Freeze first N encoder layers by default (9)
    if args.freeze_encoder_layers > 0:
        freeze_first_n_layers(model, args.freeze_encoder_layers)

    # Mixed precision flags
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

        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        eval_steps=None if args.eval_strategy == "epoch" else args.eval_steps,
        save_steps=None if args.save_strategy == "epoch" else args.save_steps,
        save_total_limit=args.save_total_limit,

        fp16=fp16_flag,
        bf16=bf16_flag,

        dataloader_num_workers=2,
        report_to=None if args.report_to == "none" else args.report_to,
        run_name=args.run_name,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    callbacks = []
    if args.early_stopping:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        ))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dsdict["train"],
        eval_dataset=dsdict["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks if callbacks else None,
    )

    # Optional: initial evaluation
    print(">>> Evaluating before training...")
    try:
        init_metrics = trainer.evaluate()
        print(init_metrics)
    except Exception as e:
        print(f"Initial evaluation skipped: {e}")

    # Train
    print(">>> Training...")
    trainer.train()

    # Final evaluation
    print(">>> Evaluating after training...")
    final_metrics = trainer.evaluate()
    print(final_metrics)

    # Save model & tokenizer
    print(f">>> Saving fine-tuned classifier to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(">>> Done.")


if __name__ == "__main__":
    main()
