import os
import json
import pandas as pd
from typing import List, Dict, Any, Union

class Dataset:
    """Class representing a dataset with texts and multi-aspect ground-truth labels."""
    
    def __init__(self, texts: List[str], aspects: Dict[str, List[Any]]):
        """
        Initialize the dataset.
        
        Args:
            texts: List of document/review texts.
            aspects: Dictionary mapping aspect names (e.g. 'Sentiment', 'Subject') to lists of labels.
        """
        self.texts = texts
        self.aspects = aspects
        
    def get_texts(self) -> List[str]:
        return self.texts
        
    def get_aspect_labels(self, aspect: str) -> List[Any]:
        if aspect not in self.aspects:
            raise ValueError(f"Aspect '{aspect}' not found in dataset. Available: {list(self.aspects.keys())}")
        return self.aspects[aspect]
        
    def __len__(self) -> int:
        return len(self.texts)

def load_dataset(filepath: str) -> Dataset:
    """
    Load a dataset from a CSV or JSON file.
    Expects a column/field named 'text' or 'content' for the texts.
    Other columns/fields are treated as aspect labels.
    
    Args:
        filepath: Path to the dataset file.
        
    Returns:
        A Dataset object.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
        
    _, ext = os.path.splitext(filepath.lower())
    
    if ext == '.csv':
        df = pd.read_csv(filepath)
        # Find text column
        text_cols = [c for c in df.columns if c.lower() in ('text', 'content')]
        if not text_cols:
            raise ValueError("CSV must contain a 'text' or 'content' column.")
        text_col = text_cols[0]
        
        texts = df[text_col].astype(str).tolist()
        
        aspects = {}
        for col in df.columns:
            if col != text_col:
                aspects[col] = df[col].tolist()
                
    elif ext == '.json':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Support either list of dicts or dict of lists
        if isinstance(data, list):
            if not data:
                raise ValueError("JSON array is empty.")
            # List of dicts
            text_key = None
            for key in ('text', 'content'):
                if key in data[0]:
                    text_key = key
                    break
            if not text_key:
                # Fallback to the first key if text/content not found
                text_key = list(data[0].keys())[0]
                
            texts = [str(item[text_key]) for item in data]
            
            aspects = {}
            for col in data[0].keys():
                if col != text_key:
                    aspects[col] = [item.get(col) for item in data]
        elif isinstance(data, dict):
            # Dict of lists
            text_key = None
            for key in ('text', 'content'):
                if key in data:
                    text_key = key
                    break
            if not text_key:
                raise ValueError("JSON object must contain a 'text' or 'content' key mapping to a list of texts.")
                
            texts = [str(t) for t in data[text_key]]
            aspects = {}
            for col, val_list in data.items():
                if col != text_key and isinstance(val_list, list):
                    aspects[col] = val_list
        else:
            raise ValueError("Unsupported JSON structure. Must be a list of objects or a dict of lists.")
    elif ext == '.jsonl':
        texts = []
        aspects = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if first_line:
                try:
                    first_item = json.loads(first_line.strip())
                    text_key = None
                    for key in ('input', 'text', 'content'):
                        if key in first_item:
                            text_key = key
                            break
                    if not text_key:
                        text_key = list(first_item.keys())[0]
                except Exception:
                    raise ValueError("JSONL file is not valid or empty.")
                
                f.seek(0)
                for line in f:
                    line_str = line.strip()
                    if line_str:
                        try:
                            item = json.loads(line_str)
                            texts.append(str(item[text_key]))
                            for col, val in item.items():
                                if col != text_key:
                                    if col not in aspects:
                                        aspects[col] = []
                                    aspects[col].append(val)
                        except Exception:
                            continue
    else:
        raise ValueError(f"Unsupported file format: {ext}. Only CSV, JSON, and JSONL are supported.")
        
    return Dataset(texts, aspects)
