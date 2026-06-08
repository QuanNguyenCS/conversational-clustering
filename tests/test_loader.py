import os
import tempfile
import pytest
from src.dataset.loader import load_dataset

def test_load_jsonl():
    content = (
        '{"task": "test", "input": "Hello", "label": "L1"}\n'
        '{"task": "test", "input": "World", "label": "L2"}\n'
    )
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w", encoding="utf-8") as temp:
        temp.write(content)
        temp_path = temp.name

    try:
        dataset = load_dataset(temp_path)
        assert len(dataset) == 2
        assert dataset.get_texts() == ["Hello", "World"]
        assert dataset.get_aspect_labels("label") == ["L1", "L2"]
        assert dataset.get_aspect_labels("task") == ["test", "test"]
    finally:
        os.remove(temp_path)
