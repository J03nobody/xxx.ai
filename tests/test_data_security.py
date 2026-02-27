import pytest
from src.data import DataManager
from unittest.mock import patch, MagicMock

def test_data_manager_allowed_dataset():
    # Mock load_dataset to avoid actual network calls
    with patch('src.data.load_dataset') as mock_load:
        with patch('src.data.AutoTokenizer.from_pretrained') as mock_tokenizer:
            mock_tokenizer.return_value.pad_token = None
            mock_tokenizer.return_value.eos_token = '<eos>'

            # Should succeed
            dm = DataManager(dataset_name="wikitext")
            assert dm.dataset is not None

def test_data_manager_disallowed_dataset():
    with patch('src.data.load_dataset') as mock_load:
        with patch('src.data.AutoTokenizer.from_pretrained') as mock_tokenizer:
             # Should fail
            with pytest.raises(ValueError, match="is not allowed"):
                DataManager(dataset_name="malicious/dataset")

def test_data_manager_tinystories_allowed():
    with patch('src.data.load_dataset') as mock_load:
        with patch('src.data.AutoTokenizer.from_pretrained') as mock_tokenizer:
            mock_tokenizer.return_value.pad_token = None
            mock_tokenizer.return_value.eos_token = '<eos>'

            # Should succeed
            dm = DataManager(dataset_name="roneneldan/TinyStories")
            assert dm.dataset is not None
