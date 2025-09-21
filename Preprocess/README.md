# Preprocess

Text cleaning utilities for maritime and finance news data.

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

### Clean single text

```python
from cleaning import TextCleaner

cleaner = TextCleaner()
text = "Captain John Smith\ncontacted us at john@shipping.com about vessel IMO1234567."
cleaned_text = cleaner.clean_text(text)
```

### Clean DataFrame

```python
import pandas as pd
from cleaning import TextCleaner

df = pd.read_csv("your_data.csv")
cleaner = TextCleaner()
cleaned_df = cleaner.clean_dataframe(df, 'text_column')
cleaned_df.to_csv("cleaned_data.csv", index=False)
```

### Custom cleaning options

```python
cleaned_text = cleaner.clean_text(
    text,
    remove_newlines=True,
    remove_emails=True,
    remove_urls=True,
    remove_persons=False,  # Skip person name removal
    remove_vessels=True
)
```
