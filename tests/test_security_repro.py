import unittest
from unittest.mock import patch
from src.data import DataManager

class TestSecurityFix(unittest.TestCase):
    @patch('src.data.load_dataset')
    @patch('src.data.AutoTokenizer.from_pretrained')
    def test_rejection_of_arbitrary_tokenizer(self, mock_from_pretrained, mock_load_dataset):
        # Setup mocks
        mock_tokenizer = mock_from_pretrained.return_value
        mock_tokenizer.pad_token = None

        # Test Case: Initialize DataManager with a potentially malicious tokenizer name
        malicious_input = "malicious-repo/pwned-tokenizer"

        with self.assertRaises(ValueError) as context:
            dm = DataManager(tokenizer_name=malicious_input)

        self.assertIn("Tokenizer 'malicious-repo/pwned-tokenizer' is not allowed", str(context.exception))

        # Verify from_pretrained was NOT called
        mock_from_pretrained.assert_not_called()

    @patch('src.data.load_dataset')
    @patch('src.data.AutoTokenizer.from_pretrained')
    def test_acceptance_of_allowed_tokenizer(self, mock_from_pretrained, mock_load_dataset):
        # Setup mocks
        mock_tokenizer = mock_from_pretrained.return_value
        mock_tokenizer.pad_token = None

        allowed_input = "gpt2"

        try:
            dm = DataManager(tokenizer_name=allowed_input)
        except ValueError:
            self.fail(f"DataManager rejected allowed tokenizer: {allowed_input}")

        # Verify from_pretrained was called with trust_remote_code=False
        mock_from_pretrained.assert_called_with(allowed_input, trust_remote_code=False)

if __name__ == '__main__':
    unittest.main()
