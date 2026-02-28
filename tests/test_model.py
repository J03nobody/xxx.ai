import torch
import pytest
from src.model import GPTLanguageModel

def test_gpt_language_model_initialization():
    vocab_size = 100
    block_size = 32
    n_embd = 64
    n_head = 4
    n_layer = 2

    model = GPTLanguageModel(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer)

    assert model.block_size == block_size
    assert model.token_embedding_table.weight.shape == (vocab_size, n_embd)
    assert model.position_embedding_table.weight.shape == (block_size, n_embd)
    assert len(model.blocks) == n_layer
    assert model.lm_head.weight.shape == (vocab_size, n_embd)

def test_gpt_language_model_forward_pass():
    vocab_size = 100
    block_size = 32
    n_embd = 64
    n_head = 4
    n_layer = 2
    batch_size = 4

    model = GPTLanguageModel(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer)

    # Create dummy input
    idx = torch.randint(0, vocab_size, (batch_size, block_size))

    # Forward pass without targets
    logits, loss = model(idx)
    assert logits.shape == (batch_size, block_size, vocab_size)
    assert loss is None

    # Forward pass with targets
    logits, loss = model(idx, targets=idx)
    # The current implementation of forward flattens logits to (B*T, C) when targets are provided
    # B, T, C = logits.shape
    # logits = logits.view(B*T, C)
    # So the returned logits has shape (B*T, C)
    assert logits.shape == (batch_size * block_size, vocab_size)
    assert loss is not None
    assert loss.dim() == 0 # scalar

def test_gpt_language_model_generate():
    vocab_size = 100
    block_size = 32
    n_embd = 64
    n_head = 4
    n_layer = 2

    model = GPTLanguageModel(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer)

    # Start token
    idx = torch.zeros((1, 1), dtype=torch.long)
    max_new_tokens = 10

    # Generate
    out = model.generate(idx, max_new_tokens=max_new_tokens)

    assert out.shape == (1, 1 + max_new_tokens)
    assert isinstance(out, torch.Tensor)
