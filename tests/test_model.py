import torch
import pytest
from src.model import Head

def test_head_initialization():
    head_size = 16
    n_embd = 32
    block_size = 8
    head = Head(head_size, n_embd, block_size)

    assert isinstance(head.key, torch.nn.Linear)
    assert isinstance(head.query, torch.nn.Linear)
    assert isinstance(head.value, torch.nn.Linear)
    assert hasattr(head, 'tril')
    assert head.tril.shape == (block_size, block_size)

def test_head_forward_shape():
    head_size = 16
    n_embd = 32
    block_size = 8
    batch_size = 4
    time_steps = 8

    head = Head(head_size, n_embd, block_size)
    x = torch.randn(batch_size, time_steps, n_embd)
    output = head(x)

    assert output.shape == (batch_size, time_steps, head_size)

def test_head_causal_masking():
    head_size = 16
    n_embd = 32
    block_size = 8
    batch_size = 1
    time_steps = 4

    head = Head(head_size, n_embd, block_size)
    head.eval() # Turn off dropout for deterministic behavior

    x = torch.randn(batch_size, time_steps, n_embd)

    # Run forward pass
    output1 = head(x)

    # Modify the last token in input
    x_modified = x.clone()
    x_modified[:, -1, :] = torch.randn(1, n_embd)

    output2 = head(x_modified)

    # The output at t=0, t=1, t=2 should be identical because they shouldn't attend to t=3
    # However, output at t=3 will change.

    # Check that output up to last step is same
    assert torch.allclose(output1[:, :-1, :], output2[:, :-1, :], atol=1e-6)

    # Check that the last step is different (highly likely)
    assert not torch.allclose(output1[:, -1, :], output2[:, -1, :], atol=1e-6)

def test_head_dropout():
    head_size = 16
    n_embd = 32
    block_size = 8
    head = Head(head_size, n_embd, block_size)
    x = torch.randn(2, 4, n_embd)

    # Train mode (dropout active)
    head.train()
    out1 = head(x)
    out2 = head(x)
    assert not torch.allclose(out1, out2), "Dropout should make outputs different in train mode"

    # Eval mode (dropout inactive)
    head.eval()
    out3 = head(x)
    out4 = head(x)
    assert torch.allclose(out3, out4), "Dropout should be inactive in eval mode"
