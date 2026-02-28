import unittest
import torch
from src.model import GPTLanguageModel

class TestGPTLanguageModel(unittest.TestCase):
    def setUp(self):
        self.model = GPTLanguageModel(
            vocab_size=100,
            block_size=8,
            n_embd=16,
            n_head=2,
            n_layer=2
        )

    def test_forward_pass(self):
        x = torch.randint(0, 100, (2, 8))
        logits, loss = self.model(x)
        self.assertEqual(logits.shape, (2, 8, 100))
        self.assertIsNone(loss)

    def test_forward_pass_with_targets(self):
        x = torch.randint(0, 100, (2, 8))
        y = torch.randint(0, 100, (2, 8))
        logits, loss = self.model(x, y)
        self.assertEqual(logits.shape, (16, 100)) # Reshaped for loss calculation
        self.assertIsNotNone(loss)

    def test_generate(self):
        x = torch.zeros((1, 1), dtype=torch.long)
        y = self.model.generate(x, max_new_tokens=5)
        self.assertEqual(y.shape, (1, 6))

if __name__ == '__main__':
    unittest.main()
