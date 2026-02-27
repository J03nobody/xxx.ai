import unittest
from unittest.mock import MagicMock, patch
import torch
import sys
import os

# Ensure src can be imported
sys.path.append(os.getcwd())

from src.data import DataManager

class TestDataManager(unittest.TestCase):
    @patch('src.data.load_dataset')
    @patch('src.data.AutoTokenizer')
    def test_get_batch_infinite_loop_prevention(self, mock_tokenizer, mock_load_dataset):
        # Mock dataset that yields short items infinitely
        mock_dataset = MagicMock()
        def infinite_generator():
            while True:
                yield {'text': 'short'}
        mock_dataset.__iter__.side_effect = infinite_generator
        mock_load_dataset.return_value = mock_dataset

        # Mock tokenizer to produce short sequences
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.encode.return_value = [1] # length 1
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Initialize DataManager with block_size=10 (so encoded len 1 is filtered out)
        dm = DataManager(block_size=10, batch_size=1)

        # Expect RuntimeError due to max attempts exceeded
        with self.assertRaises(RuntimeError) as cm:
            dm.get_batch()

        self.assertIn("Unable to fill batch", str(cm.exception))

    @patch('src.data.load_dataset')
    @patch('src.data.AutoTokenizer')
    def test_get_batch_empty_dataset(self, mock_tokenizer, mock_load_dataset):
        # Mock dataset that is empty
        mock_dataset = MagicMock()
        def empty_generator():
            return
            yield
        mock_dataset.__iter__.side_effect = empty_generator
        mock_load_dataset.return_value = mock_dataset

        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        dm = DataManager(batch_size=1)

        # Expect RuntimeError due to empty dataset
        with self.assertRaises(RuntimeError) as cm:
            dm.get_batch()

        self.assertIn("Dataset is empty", str(cm.exception))

if __name__ == '__main__':
    unittest.main()
