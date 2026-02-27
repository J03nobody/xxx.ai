import torch
import pytest
from src.model import GPTLanguageModel, Head, MultiHeadAttention, Block, FeedFoward

def test_gpt_model_initialization():
    vocab_size = 100
    block_size = 32
    n_embd = 64
    n_head = 4
    n_layer = 2

    model = GPTLanguageModel(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer)

    assert model.token_embedding_table.weight.shape == (vocab_size, n_embd)
    assert model.position_embedding_table.weight.shape == (block_size, n_embd)
    assert len(model.blocks) == n_layer

def test_gpt_forward_pass():
    vocab_size = 100
    block_size = 32
    n_embd = 64
    n_head = 4
    n_layer = 2
    batch_size = 4

    model = GPTLanguageModel(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer)

    # Create dummy input
    idx = torch.randint(0, vocab_size, (batch_size, block_size))

    logits, loss = model(idx)

    assert logits.shape == (batch_size, block_size, vocab_size)
    assert loss is None

    # Test with targets
    targets = torch.randint(0, vocab_size, (batch_size, block_size))
    logits, loss = model(idx, targets)

    assert loss is not None
    assert isinstance(loss.item(), float)

def test_causal_masking():
    block_size = 8
    n_embd = 16
    head_size = 4

    head = Head(head_size, n_embd, block_size)

    # Check tril buffer
    assert hasattr(head, 'tril')
    assert head.tril.shape == (block_size, block_size)
    assert torch.equal(head.tril, torch.tril(torch.ones(block_size, block_size)))

def test_block_forward():
    n_embd = 32
    n_head = 4
    block_size = 16
    batch_size = 2

    block = Block(n_embd, n_head, block_size)
    x = torch.randn(batch_size, block_size, n_embd)

    out = block(x)
    assert out.shape == (batch_size, block_size, n_embd)
