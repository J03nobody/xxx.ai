
import pytest
import torch
from unittest.mock import MagicMock
from src.data import DataManager

class TestDataManager:
    def test_get_batch_empty_dataset(self):
        """Test that get_batch raises RuntimeError when the dataset is empty."""
        dm = DataManager()
        # Mock dataset to be empty
        dm.dataset = []
        dm.dataset_iterator = iter([])
        dm.tokenizer = MagicMock()
        dm.tokenizer.encode.return_value = []

        with pytest.raises(RuntimeError, match="Dataset is empty"):
            dm.get_batch()

    def test_get_batch_filtered_dataset(self):
        """Test that get_batch raises RuntimeError when all data is filtered out."""
        dm = DataManager()
        # Mock dataset with items that are too short
        dm.dataset = [{'text': 'short'}] * 2000
        dm.dataset_iterator = iter(dm.dataset)
        dm.tokenizer = MagicMock()
        # Return short encoding
        dm.tokenizer.encode.return_value = [1]
        dm.block_size = 10

        with pytest.raises(RuntimeError, match="Unable to find valid data"):
             dm.get_batch()
