# Pipeline

A complete pipeline for maritime and finance news text classification using RoBERTa.

## Workflow

### 1. Preprocessing
First, collect your data from the maritime data source with stable publication frequency (e.g., weekly articles or 5+ articles per month), and check data usage agreements to ensure research compliance.

Suggested data sources:
TradeWinds, Marinelink, Hellenic, Seatrade Maritime, Dry Bulk, Lloyd's List, Clarkson

Then clean your text data using the preprocessing utilities in the `Preprocess/` folder.

### 2. Model Training
Choose one of the following approaches:

#### Option A: Python Scripts (Shipping_Roberta)
Use the Python scripts for command-line training with full control.

#### Option B: Jupyter Notebook (Shipping_Roberta_ipynb)
Use the Jupyter notebook for interactive training and experimentation.

## Project Structure

```
Pipeline/
├── Preprocess/                    # Text cleaning utilities
│   ├── cleaning.py               # Main cleaning script
│   └── requirements.txt          # Dependencies
├── Shipping_Roberta/             # Python script version
│   ├── dapt/train_dapt.py        # Domain-adaptive pre-training
│   ├── tapt/train_tapt.py        # Task-adaptive pre-training  
│   ├── train_classifier.py       # Classifier fine-tuning
│   └── Data/                     # Input data files
└── Shipping_Roberta_ipynb/       # Jupyter notebook version
    ├── Shipping_Roberta Pipeline.ipynb  # Complete pipeline
    └── Data/                     # Input data files
```

## Data Requirements

- **maritime_corpus.csv**: Maritime domain text for DAPT
- **News_corpus.csv**: News articles for TAPT  
- **Annotation.csv**: Labeled sentences for classifier training

## Quick Start

1. **Preprocess** your raw text data
2. **Choose** Python scripts OR Jupyter notebook
3. **Train** models sequentially: DAPT → TAPT → Classifier
4. **Evaluate** results and deploy
