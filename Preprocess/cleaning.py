#!/usr/bin/env python3
"""
Text Cleaning Script for Maritime and Finance News
Cleans text by removing newlines, emails, URLs, person names (NER), and vessel UIDs
"""

import re
import spacy
import pandas as pd
from typing import List, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class TextCleaner:
    """
    A comprehensive text cleaning class for maritime and finance news data.
    """
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the text cleaner with spaCy model for NER.
        
        Args:
            spacy_model: spaCy model name for NER (default: en_core_web_sm)
        """
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"Warning: spaCy model '{spacy_model}' not found.")
            print("Please install it with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def remove_newlines(self, text: str) -> str:
        """
        Remove newline characters and normalize whitespace.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text with newlines removed
        """
        # Replace various newline characters with spaces
        text = re.sub(r'[\r\n\t]+', ' ', text)
        # Normalize multiple spaces to single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def remove_emails(self, text: str) -> str:
        """
        Remove email addresses from text.
        
        Args:
            text: Input text string
            
        Returns:
            Text with email addresses removed
        """
        # Email regex pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)
    
    def remove_urls(self, text: str) -> str:
        """
        Remove web URLs from text.
        
        Args:
            text: Input text string
            
        Returns:
            Text with URLs removed
        """
        # URL regex pattern (covers http, https, www, and common domains)
        url_pattern = r'(https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)'
        return re.sub(url_pattern, '', text)
    
    def remove_person_names(self, text: str) -> str:
        """
        Remove person names using Named Entity Recognition (NER).
        
        Args:
            text: Input text string
            
        Returns:
            Text with person names removed
        """
        if self.nlp is None:
            print("Warning: NER not available, skipping person name removal")
            return text
        
        doc = self.nlp(text)
        # Get all PERSON entities
        person_entities = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        
        # Remove person names from text
        cleaned_text = text
        for person in person_entities:
            cleaned_text = cleaned_text.replace(person, '')
        
        return cleaned_text
    
    def remove_vessel_uids(self, text: str) -> str:
        """
        Remove shipping vessel UIDs based on common maritime vessel identification patterns.
        
        Args:
            text: Input text string
            
        Returns:
            Text with vessel UIDs removed
        """
        # Common vessel UID patterns
        patterns = [
            # IMO numbers (7 digits)
            r'\bIMO\s*\d{7}\b',
            r'\b\d{7}\b(?=\s*(?:IMO|vessel|ship))',
            
            # MMSI numbers (9 digits)
            r'\bMMSI\s*\d{9}\b',
            r'\b\d{9}\b(?=\s*(?:MMSI|vessel|ship))',
            
            # Call signs (3-7 alphanumeric characters)
            r'\b[A-Z0-9]{3,7}\b(?=\s*(?:call\s*sign|callsign|vessel|ship))',
            
            # Vessel names in quotes or brackets
            r'"[A-Z0-9\s-]+"(?=\s*(?:vessel|ship|tanker|container))',
            r'\[[A-Z0-9\s-]+\](?=\s*(?:vessel|ship|tanker|container))',
            
            # Common vessel prefixes
            r'\b(?:MV|MS|SS|RV|FV)\s+[A-Z0-9\s-]+\b',
            
            # Container numbers (11 characters: 4 letters + 7 digits)
            r'\b[A-Z]{4}\d{7}\b',
            
            # Bill of lading numbers (various formats)
            r'\b[A-Z]{2,4}\d{6,12}\b(?=\s*(?:B/L|bill|of|lading))',
        ]
        
        cleaned_text = text
        for pattern in patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
        
        return cleaned_text
    
    def clean_text(self, text: str, remove_newlines: bool = True, 
                   remove_emails: bool = True, remove_urls: bool = True,
                   remove_persons: bool = True, remove_vessels: bool = True) -> str:
        """
        Apply all cleaning operations to the input text.
        
        Args:
            text: Input text string
            remove_newlines: Whether to remove newlines
            remove_emails: Whether to remove email addresses
            remove_urls: Whether to remove URLs
            remove_persons: Whether to remove person names
            remove_vessels: Whether to remove vessel UIDs
            
        Returns:
            Fully cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        cleaned_text = text
        
        if remove_newlines:
            cleaned_text = self.remove_newlines(cleaned_text)
        
        if remove_emails:
            cleaned_text = self.remove_emails(cleaned_text)
        
        if remove_urls:
            cleaned_text = self.remove_urls(cleaned_text)
        
        if remove_persons:
            cleaned_text = self.remove_person_names(cleaned_text)
        
        if remove_vessels:
            cleaned_text = self.remove_vessel_uids(cleaned_text)
        
        # Final cleanup: normalize whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text
    
    def clean_dataframe(self, df: pd.DataFrame, text_column: str, 
                       output_column: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Clean text data in a pandas DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of the column containing text to clean
            output_column: Name of the output column (default: text_column + '_cleaned')
            **kwargs: Additional arguments passed to clean_text()
            
        Returns:
            DataFrame with cleaned text column
        """
        if output_column is None:
            output_column = f"{text_column}_cleaned"
        
        df = df.copy()
        df[output_column] = df[text_column].apply(
            lambda x: self.clean_text(str(x), **kwargs)
        )
        
        return df


def main():
    """
    Example usage and testing of the TextCleaner class.
    """
    # Initialize the cleaner
    cleaner = TextCleaner()
    
    # Example texts with various elements to clean
    test_texts = [
        "Captain John Smith\ncontacted us at john.smith@shipping.com about vessel IMO1234567.",
        "Visit our website https://www.maritime-news.com for updates on MV Ocean Star.",
        "The container MSCU1234567 arrived at port. Contact info@port.com for details.",
        "MMSI123456789 reported engine failure. Captain Mary Johnson is investigating.",
        "Freight rates rose 15% according to https://shipping-data.org/news.",
        "Vessel 'ATLANTIC STAR' (IMO9876543) departed Singapore yesterday.",
    ]
    
    print("Text Cleaning Examples:")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nExample {i}:")
        print(f"Original: {text}")
        cleaned = cleaner.clean_text(text)
        print(f"Cleaned:  {cleaned}")
        print("-" * 30)
    
    # Example with DataFrame
    print("\nDataFrame Cleaning Example:")
    print("=" * 50)
    
    sample_data = {
        'sentence': [
            "Captain Smith\nreported vessel IMO1234567 status via email@shipping.com",
            "Visit https://maritime-news.com for updates on MV Ocean Star",
            "Container MSCU1234567 arrived. Contact info@port.com for details."
        ]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = cleaner.clean_dataframe(df, 'sentence')
    print("\nCleaned DataFrame:")
    print(cleaned_df)


if __name__ == "__main__":
    main()
