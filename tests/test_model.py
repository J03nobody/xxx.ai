import torch
import torch.nn as nn
from src.model import FeedFoward

def test_feed_forward_initialization():
    n_embd = 128
    model = FeedFoward(n_embd)

    # Check if the network is a Sequential container
    assert isinstance(model.net, nn.Sequential)

    # Check layer structure: Linear -> ReLU -> Linear -> Dropout
    assert len(model.net) == 4
    assert isinstance(model.net[0], nn.Linear)
    assert isinstance(model.net[1], nn.ReLU)
    assert isinstance(model.net[2], nn.Linear)
    assert isinstance(model.net[3], nn.Dropout)

    # Check dimensions
    assert model.net[0].in_features == n_embd
    assert model.net[0].out_features == 4 * n_embd
    assert model.net[2].in_features == 4 * n_embd
    assert model.net[2].out_features == n_embd

def test_feed_forward_forward_pass():
    batch_size = 2
    seq_len = 10
    n_embd = 32

    model = FeedFoward(n_embd)
    x = torch.randn(batch_size, seq_len, n_embd)

    output = model(x)

    # Check output shape
    assert output.shape == (batch_size, seq_len, n_embd)

    # Check that output is not NaN
    assert not torch.isnan(output).any()

def test_feed_forward_computation():
    # Simple deterministic test
    n_embd = 4
    model = FeedFoward(n_embd)

    # Set weights to known values for deterministic output testing if needed,
    # but for now we just verify it runs without error on simple inputs
    x = torch.ones(1, 1, n_embd)
    output = model(x)

    assert output.shape == (1, 1, n_embd)
    assert torch.is_tensor(output)
