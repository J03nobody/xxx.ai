import torch
import pytest
from src.model import GPTLanguageModel
from src.data import DataManager

def test_model_initialization():
    vocab_size = 100
    block_size = 32
    n_embd = 32
    n_head = 4
    n_layer = 2

    model = GPTLanguageModel(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer)
    assert isinstance(model, GPTLanguageModel)

    # Check if parameters are initialized
    assert len(list(model.parameters())) > 0

def test_model_forward_pass():
    vocab_size = 100
    block_size = 32
    n_embd = 32
    n_head = 4
    n_layer = 2
    batch_size = 4

    model = GPTLanguageModel(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer)

    # Create dummy input
    idx = torch.randint(0, vocab_size, (batch_size, block_size))

    # Forward pass
    logits, loss = model(idx)

    assert logits.shape == (batch_size, block_size, vocab_size)
    assert loss is None

def test_model_training_step():
    vocab_size = 100
    block_size = 32
    n_embd = 32
    n_head = 4
    n_layer = 2
    batch_size = 4

    model = GPTLanguageModel(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer)

    # Create dummy input and targets
    idx = torch.randint(0, vocab_size, (batch_size, block_size))
    targets = torch.randint(0, vocab_size, (batch_size, block_size))

    # Forward pass with targets
    logits, loss = model(idx, targets)

    assert loss is not None
    assert isinstance(loss.item(), float)

def test_data_manager_initialization():
    # Mocking load_dataset would be ideal, but for integration testing we can check if it initializes
    # We use a dummy name if we don't want to download data, but here we want to test the actual class logic.
    # We can skip the actual dataset loading part for unit testing by mocking, or use a very small subset.
    pass
