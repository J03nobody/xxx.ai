import torch
import sys
import os

# Add the current directory to sys.path to ensure we can import from src
sys.path.append(os.getcwd())

from src.model import GPTLanguageModel
from src.data import DataManager
from src.engine import EvolutionEngine

def test_health():
    print("Testing src.model.GPTLanguageModel...")
    vocab_size = 100
    block_size = 32
    n_embd = 32
    n_head = 4
    n_layer = 2

    model = GPTLanguageModel(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer)

    # Create dummy input
    idx = torch.randint(0, vocab_size, (2, block_size))

    # Test forward pass
    logits, loss = model(idx)
    assert logits.shape == (2, block_size, vocab_size)
    assert loss is None

    # Test forward pass with targets
    logits, loss = model(idx, idx)
    assert loss is not None

    print("Testing src.data.DataManager...")
    # Mocking DataManager instantiation to avoid downloading data if possible,
    # but here we just instantiate it.
    # NOTE: This might try to download wikitext.
    # We can try to mock load_dataset if we want to be safe, but let's see if it works or fails fast.
    # Actually, let's just check if the class exists and can be imported (already done).
    # We won't run get_batch to avoid network calls in this simple health check unless necessary.

    print("Testing src.engine.EvolutionEngine...")
    engine = EvolutionEngine(population_size=2, vocab_size=vocab_size, block_size=block_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer, device='cpu')
    assert len(engine.population) == 2

    print("Health check passed!")

if __name__ == "__main__":
    test_health()
