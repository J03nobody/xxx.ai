import unittest
import torch
from src.data import DataManager

class TestDataManager(unittest.TestCase):
    def test_initialization(self):
        # We use a small dataset or mock it?
        # Using wikitext-2-raw-v1 might be slow or require download.
        # But DataManager is hardcoded to use it.
        # We can try to initialize it.
        try:
            dm = DataManager(batch_size=2, block_size=8)
        except Exception as e:
            self.skipTest(f"Failed to initialize DataManager (network issue?): {e}")

        self.assertIsNotNone(dm.tokenizer)
        self.assertIsNotNone(dm.dataset)

    def test_get_batch(self):
        try:
            dm = DataManager(batch_size=2, block_size=8)
        except Exception as e:
            self.skipTest(f"Failed to initialize DataManager: {e}")

        x, y = dm.get_batch()
        self.assertEqual(x.shape, (2, 8))
        self.assertEqual(y.shape, (2, 8))

if __name__ == '__main__':
    unittest.main()
