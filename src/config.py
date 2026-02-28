import os
import torch
from dataclasses import dataclass

@dataclass
class GPTConfig:
    batch_size: int = 32
    block_size: int = 64
    max_iters: int = 5000
    eval_interval: int = 500
    learning_rate: float = 3e-4
    eval_iters: int = 200
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.2
    vocab_size: int = 50304 # GPT-2 vocab size roughly
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __post_init__(self):
        # Override with environment variables if present
        if os.environ.get('BATCH_SIZE'):
            self.batch_size = int(os.environ['BATCH_SIZE'])
        if os.environ.get('BLOCK_SIZE'):
            self.block_size = int(os.environ['BLOCK_SIZE'])
        if os.environ.get('MAX_ITERS'):
            self.max_iters = int(os.environ['MAX_ITERS'])
        if os.environ.get('EVAL_INTERVAL'):
            self.eval_interval = int(os.environ['EVAL_INTERVAL'])
        if os.environ.get('LEARNING_RATE'):
            self.learning_rate = float(os.environ['LEARNING_RATE'])
        if os.environ.get('EVAL_ITERS'):
            self.eval_iters = int(os.environ['EVAL_ITERS'])
        if os.environ.get('N_EMBD'):
            self.n_embd = int(os.environ['N_EMBD'])
        if os.environ.get('N_HEAD'):
            self.n_head = int(os.environ['N_HEAD'])
        if os.environ.get('N_LAYER'):
            self.n_layer = int(os.environ['N_LAYER'])
        if os.environ.get('DROPOUT'):
            self.dropout = float(os.environ['DROPOUT'])
        if os.environ.get('VOCAB_SIZE'):
            self.vocab_size = int(os.environ['VOCAB_SIZE'])
        if os.environ.get('DEVICE'):
            self.device = os.environ['DEVICE']
