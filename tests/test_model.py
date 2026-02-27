import unittest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock
from src.model import Block, MultiHeadAttention, FeedFoward

class TestBlock(unittest.TestCase):
    def test_initialization(self):
        n_embd = 32
        n_head = 4
        block_size = 16
        block = Block(n_embd, n_head, block_size)

        self.assertIsInstance(block.sa, MultiHeadAttention)
        self.assertIsInstance(block.ffwd, FeedFoward)
        self.assertIsInstance(block.ln1, nn.LayerNorm)
        self.assertIsInstance(block.ln2, nn.LayerNorm)

    def test_forward_pass_shape(self):
        n_embd = 32
        n_head = 4
        block_size = 16
        block = Block(n_embd, n_head, block_size)

        # Input tensor of shape (Batch, Time, Channels)
        x = torch.randn(2, 10, 32)
        out = block(x)

        self.assertEqual(out.shape, x.shape)

    def test_forward_pass_logic(self):
        n_embd = 32
        n_head = 4
        block_size = 16
        block = Block(n_embd, n_head, block_size)

        # Create a mock that inherits from nn.Module to satisfy torch's type checking
        class MockModule(nn.Module):
            def __init__(self, side_effect):
                super().__init__()
                self.side_effect = side_effect
                self.called = False

            def forward(self, x):
                self.called = True
                return self.side_effect(x)

        # Mock self-attention and feed-forward layers to return zeros
        mock_sa = MockModule(side_effect=lambda x: torch.zeros_like(x))
        mock_ffwd = MockModule(side_effect=lambda x: torch.zeros_like(x))

        block.sa = mock_sa
        block.ffwd = mock_ffwd

        x = torch.ones(2, 10, 32)
        out = block(x)

        # Check residual connection: output should equal input since mocked components return 0
        self.assertTrue(torch.allclose(out, x), "Residual connection failed: output does not match input with zeroed sub-layers")

        # Verify that sub-layers were called
        self.assertTrue(mock_sa.called)
        self.assertTrue(mock_ffwd.called)
