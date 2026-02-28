import torch
import unittest
from src.engine import EvolutionEngine

class TestEvolutionEngine(unittest.TestCase):
    def setUp(self):
        self.engine = EvolutionEngine(
            population_size=2,
            vocab_size=100,
            block_size=8,
            n_embd=16,
            n_head=2,
            n_layer=2,
            device='cpu'
        )

    def test_initialization(self):
        self.assertEqual(len(self.engine.population), 2)
        self.assertEqual(len(self.engine.optimizers), 2)

    def test_train_step(self):
        x = torch.randint(0, 100, (4, 8))
        y = torch.randint(0, 100, (4, 8))
        loss = self.engine.train_step(x, y, 0)
        self.assertIsInstance(loss, float)

    def test_evaluate(self):
        x = torch.randint(0, 100, (4, 8))
        y = torch.randint(0, 100, (4, 8))
        loss = self.engine.evaluate(0, x, y)
        self.assertIsInstance(loss, float)

    def test_evolve(self):
        val_losses = [1.5, 0.5]
        best_idx = self.engine.evolve(val_losses)
        self.assertEqual(best_idx, 1)
        # Check if population is updated (e.g., all models should be based on the best model)
        # We can check if the weights of model 0 are different from what they were initially,
        # or if they are now close to model 1 (but mutated).
        # Since we don't have the previous state easily accessible here without more complex setup,
        # we just ensure the method runs without error and returns the correct index.

if __name__ == '__main__':
    unittest.main()
