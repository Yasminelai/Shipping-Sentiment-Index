# Shipping_Roberta

<details>
<summary>Project Structure</summary>

```text
Shipping_Roberta/
├─ dapt/
│  └─ train_dapt.py # Domain-Adaptive Pre-Training (MLM) on maritime_corpus.csv
├─ tapt/
│  └─ train_tapt.py # Task-Adaptive Pre-Training (MLM) on News_corpus.csv,
│                   # initialized from the DAPT checkpoint
├─ train_classifier.py # Fine-tunes the TAPT model on Annotation.csv for classification
├─ Data/ # All input data (already cleaned)
│  ├─ maritime_corpus.csv # Column: sentence; used for DAPT
│  ├─ News_corpus.csv # Columns: date, url, title, text; only text is used for TAPT
│  └─ Annotation.csv # Columns: sentence, label; used to train the classifier
└─ Model/ # Saved model checkpoints
   ├─ dapt_model/ # Output from DAPT training (HF model + tokenizer)
   ├─ tapt_model/ # Output from TAPT training (initialized from dapt_model)
   └─ classifier/ # Output from classifier fine-tuning
```
</details>

## Overview
This project builds a **RoBERTa-based text classification system** for maritime and finance news.

- **DAPT** – Domain-adaptive pre-training on `maritime_corpus.csv` to adapt RoBERTa to maritime language.
- **TAPT** – Task-adaptive pre-training on the news corpus, continuing from the DAPT checkpoint to match task-specific news style.
- **Classifier** – Supervised fine-tuning using `Annotation.csv` to predict labels for cleaned sentences.
- **Model Folder** – Stores Hugging Face–style checkpoints for each training stage, ready for reuse or deployment.

## Requirements
- Python 3.10+
- `transformers`
- `datasets`
- `torch`

Install dependencies:
```bash 
pip install transformers datasets torch
```

## Quick start with sample data (minimal flow)

### Prerequisites
```shell
conda env create -f environment.yml
conda activate shipping-roberta
```

###Enable Weights & Biases (same pattern)
```shell
pip install wandb
wandb login
```
The following are all the training parameters used in the scripts. 
Please adjust them to suit your specific training needs.
###  DAPT on Data/maritime_corpus.csv 
```shell
python dapt/train_dapt.py ^  # Domain-adaptive pre-training (MLM)
  --data_path Data/maritime_corpus.csv ^        # Input CSV path
  --text_column sentence ^                      # Column name used as text
  --output_dir Model/dapt_model ^               # Output dir for checkpoints/tokenizer
  --model_name_or_path roberta-base ^           # HF base model to start from
  --max_length 128 ^                            # Max tokens per example (truncate)
  --mlm_probability 0.15 ^                      # Masking ratio for MLM
  --per_device_train_batch_size 16 ^            # Train batch size per device
  --per_device_eval_batch_size 16 ^             # Eval batch size per device
  --learning_rate 5e-5 ^                        # AdamW learning rate
  --weight_decay 0.01 ^                         # AdamW weight decay
  --num_train_epochs 3 ^                        # Total epochs
  --warmup_ratio 0.06 ^                         # Linear warmup ratio
  --gradient_accumulation_steps 1 ^             # Steps to accumulate before optimizer
  --eval_steps 1000 ^                           # Evaluate every N training steps
  --save_steps 1000 ^                           # Save checkpoint every N steps
  --logging_steps 100 ^                         # Log metrics every N steps
  --save_total_limit 2 ^                        # Keep at most N checkpoints
  --seed 42 ^                                   # Random seed
  --train_val_split 0.95 ^                      # Train proportion (rest for validation)
  --fp16 ^                                      # Use FP16 when available (flag)
  --bf16 ^                                      # Use BF16 on Ampere+ GPUs (flag)
  --num_proc 1 ^                                # CPU workers for dataset map
  --report_to wandb ^                           # Logging backend: none|wandb|tensorboard
  --run_name dapt_run ^                         # Experiment/run name
  --early_stopping ^                            # Enable early stopping (flag)
  --early_stopping_patience 3 ^                 # Eval rounds without improvement
  --early_stopping_threshold 0.0 ^              # Min delta to count as improvement
  --resume_from_checkpoint Model/dapt_model/checkpoint-XXXX ^  # Resume specific ckpt
  --auto_resume                                 # Auto-resume latest ckpt (flag)
```

###  TAPT on Data/News_corpus.csv, initialized from DAPT
```shell
python tapt/train_tapt.py ^  # Task-adaptive pre-training (MLM)
  --data_path Data/News_corpus.csv ^            # Input CSV path
  --text_column text ^                          # Column name used as text
  --init_from Model/dapt_model ^                # Initialize from DAPT output
  --output_dir Model/tapt_model ^               # Output dir for checkpoints/tokenizer
  --max_length 256 ^                            # Max tokens per example
  --mlm_probability 0.15 ^                      # Masking ratio for MLM
  --per_device_train_batch_size 16 ^            # Train batch size per device
  --per_device_eval_batch_size 16 ^             # Eval batch size per device
  --learning_rate 5e-5 ^                        # AdamW learning rate
  --weight_decay 0.01 ^                         # AdamW weight decay
  --num_train_epochs 3 ^                        # Total epochs
  --warmup_ratio 0.06 ^                         # Linear warmup ratio
  --gradient_accumulation_steps 1 ^             # Steps to accumulate before optimizer
  --eval_steps 1000 ^                           # Evaluate every N training steps
  --save_steps 1000 ^                           # Save checkpoint every N steps
  --logging_steps 100 ^                         # Log metrics every N steps
  --save_total_limit 2 ^                        # Keep at most N checkpoints
  --seed 42 ^                                   # Random seed
  --train_val_split 0.95 ^                      # Train proportion (rest for validation)
  --fp16 ^                                      # Use FP16 when available (flag)
  --bf16 ^                                      # Use BF16 on Ampere+ GPUs (flag)
  --num_proc 1 ^                                # CPU workers for dataset map
  --report_to wandb ^                           # Logging backend: none|wandb|tensorboard
  --run_name tapt_run ^                         # Experiment/run name
  --early_stopping ^                            # Enable early stopping (flag)
  --early_stopping_patience 3 ^                 # Eval rounds without improvement
  --early_stopping_threshold 0.0 ^              # Min delta to count as improvement
  --resume_from_checkpoint Model/tapt_model/checkpoint-XXXX ^  # Resume specific ckpt
  --auto_resume                                 # Auto-resume latest ckpt (flag)
```

###  Classifier fine-tuning on Data/Annotation.csv
```shell
python train_classifier.py ^  # Supervised fine-tuning for classification
  --data_path Data/Annotation.csv ^             # Input CSV path
  --text_column sentence ^                      # Text column name
  --label_column label ^                        # Label column name
  --tapt_dir Model/tapt_model ^                 # Initialize from TAPT output
  --output_dir Model/classifier ^               # Output dir for classifier
  --max_length 256 ^                            # Max tokens per example
  --freeze_encoder_layers 9 ^                   # Freeze first N encoder layers
  --per_device_train_batch_size 16 ^            # Train batch size per device
  --per_device_eval_batch_size 32 ^             # Eval batch size per device
  --learning_rate 2e-5 ^                        # AdamW learning rate
  --weight_decay 0.01 ^                         # AdamW weight decay
  --num_train_epochs 4 ^                        # Total epochs
  --warmup_ratio 0.06 ^                         # Linear warmup ratio
  --gradient_accumulation_steps 1 ^             # Steps to accumulate before optimizer
  --logging_steps 100 ^                         # Log metrics every N steps
  --eval_strategy epoch ^                       # Evaluate by steps|epoch
  --save_strategy epoch ^                       # Save ckpt by steps|epoch
  --eval_steps 500 ^                            # Used if eval_strategy=steps
  --save_steps 500 ^                            # Used if save_strategy=steps
  --save_total_limit 2 ^                        # Keep at most N checkpoints
  --seed 42 ^                                   # Random seed
  --train_val_split 0.9 ^                       # Train proportion (rest for validation)
  --fp16 ^                                      # Use FP16 when available (flag)
  --bf16 ^                                      # Use BF16 on Ampere+ GPUs (flag)
  --num_proc 1 ^                                # CPU workers for dataset map
  --report_to wandb ^                           # Logging backend: none|wandb|tensorboard
  --run_name clf_run ^                          # Experiment/run name
  --early_stopping ^                            # Enable early stopping (flag)
  --early_stopping_patience 2 ^                 # Eval rounds without improvement
  --early_stopping_threshold 0.0                # Min delta to count as improvement
```
