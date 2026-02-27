import os
import pytest
from src.config import GPTConfig

def test_default_config():
    config = GPTConfig()
    assert config.batch_size == 32
    assert config.block_size == 64
    assert config.learning_rate == 3e-4

def test_env_override():
    os.environ['BATCH_SIZE'] = '64'
    os.environ['BLOCK_SIZE'] = '128'
    os.environ['LEARNING_RATE'] = '1e-3'

    config = GPTConfig()
    assert config.batch_size == 64
    assert config.block_size == 128
    assert config.learning_rate == 1e-3

    # Cleanup
    del os.environ['BATCH_SIZE']
    del os.environ['BLOCK_SIZE']
    del os.environ['LEARNING_RATE']
