# Shipping_Roberta Pipeline (Jupyter Notebook Version)

<details>
<summary>Project Structure</summary>

```text
Shipping_Roberta_ipynb/
├─ Shipping_Roberta Pipeline.ipynb # Complete training pipeline in Jupyter notebook
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
This project provides a **complete Jupyter notebook pipeline** for training a RoBERTa-based text classification system for maritime and finance news.

- **DAPT** – Domain-adaptive pre-training on `maritime_corpus.csv` to adapt RoBERTa to maritime language.
- **TAPT** – Task-adaptive pre-training on the news corpus, continuing from the DAPT checkpoint to match task-specific news style.
- **Classifier** – Supervised fine-tuning using `Annotation.csv` to predict labels (irrelevant/rise/fall) for cleaned sentences.
- **Evaluation** – Comprehensive evaluation with temperature scaling and two-stage prediction.
- **Model Folder** – Stores Hugging Face–style checkpoints for each training stage, ready for reuse or deployment.

## Requirements
- Python 3.10+
- `transformers>=4.41.0`
- `datasets>=2.19.0`
- `torch`
- `scikit-learn>=1.3.0`
- `pandas>=2.0.0`

Install dependencies:
```bash 
pip install "transformers>=4.41.0" "datasets>=2.19.0" "accelerate>=0.30.0" "scikit-learn>=1.3.0" "pandas>=2.0.0" "openpyxl>=3.1.0"
```

## Quick Start

### Prerequisites
```shell
conda env create -f environment.yml
conda activate shipping-roberta
```

### Running the Pipeline
1. Open `Shipping_Roberta Pipeline.ipynb` in Jupyter
2. Run all cells sequentially:
   - **Cell 1**: Data loading functions
   - **Cell 2**: Package installation and imports
   - **Cell 3**: DAPT training
   - **Cell 4**: TAPT training  
   - **Cell 5**: Classifier data preparation
   - **Cell 6**: Classifier model definition and training
   - **Cell 7**: Evaluation and results

### Training Parameters

#### DAPT Configuration
- **Model**: roberta-base
- **Max Length**: 128 tokens
- **MLM Probability**: 0.15
- **Batch Size**: 8 (train), 16 (eval)
- **Learning Rate**: 5e-5
- **Epochs**: 2
- **Eval Steps**: 500
- **Save Steps**: 1000

#### TAPT Configuration  
- **Initialize From**: DAPT model checkpoint
- **Max Length**: 512 tokens
- **MLM Probability**: 0.15
- **Batch Size**: 16 (train), 8 (eval)
- **Learning Rate**: 3e-5
- **Epochs**: 2
- **Eval Steps**: 50
- **Save Steps**: 100

#### Classifier Configuration
- **Initialize From**: TAPT model checkpoint
- **Max Length**: 128 tokens
- **Batch Size**: 8 (train), 32 (eval)
- **Learning Rate**: 1e-5 (encoder), 3e-5 (classifier heads)
- **Epochs**: 5
- **Frozen Layers**: 9 encoder layers
- **Two-Stage Model**: Relevance + Direction classification

### Features

#### Automatic Checkpoint Management
- **Auto-resume**: Automatically resumes from latest checkpoint
- **Manual Resume**: Specify exact checkpoint path if needed
- **Checkpoint Listing**: View available checkpoints for each model

#### Advanced Evaluation
- **Two-Stage Prediction**: First relevance, then direction classification
- **Temperature Scaling**: Calibrates prediction probabilities
- **Comprehensive Metrics**: Classification report and confusion matrix
- **Threshold Optimization**: Configurable relevance threshold (default: 0.5)

#### Model Architecture
- **Two-Stage Classifier**: 
  - Stage 1: Irrelevant vs Relevant
  - Stage 2: Rise vs Fall (for relevant samples)
- **Class Weighting**: Handles imbalanced datasets
- **Layer Freezing**: Freezes early encoder layers for stability

