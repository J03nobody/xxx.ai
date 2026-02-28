# Evolutionary Language Model Training

This repository implements a Transformer-based language model (nanoGPT style) with an evolutionary training engine. Instead of relying solely on backpropagation and gradient descent on a single model, this project maintains a *population* of models, trains them in parallel, evaluates their performance, and "evolves" the best-performing models to encourage exploration and stability.

## Features

- **Transformer Architecture**: Implements a standard GPT-style architecture with Multi-Head Self-Attention, FeedForward layers, Layer Normalization, and residual connections.
- **Evolutionary Engine**: A simple evolutionary strategy where a population of models is trained. At each generation, the best-performing model's weights are cloned, slightly mutated, and used to seed the next generation.
- **Kaggle Dataset Integration**: Leverages the `kagglehub` library to automatically download public Kaggle datasets (currently defaults to a popular video games dataset) without requiring authentication.
- **Hugging Face Tokenizer**: Uses the `gpt2` tokenizer from the `transformers` library for robust text encoding and decoding.
- **Streaming Data**: Utilizes Hugging Face `datasets` for efficient, streaming data processing from local CSV files.

## Project Structure

- `src/model.py`: Contains the PyTorch implementation of the `GPTLanguageModel`, including `Head`, `MultiHeadAttention`, `FeedForward`, and `Block` classes.
- `src/data.py`: Manages data downloading from Kaggle, CSV parsing, tokenization, and batch generation.
- `src/engine.py`: Defines the `EvolutionEngine`, which handles the population of models, training steps, evaluation, and the evolutionary selection/mutation process.
- `src/main.py`: The entry point script that orchestrates the training loop over multiple generations and demonstrates text generation.

## Requirements

Ensure you have Python 3.x installed. You will need the following libraries:

```bash
pip install torch transformers datasets kagglehub
```

## How to Run

To run the evolutionary training simulation, simply execute the main script from the root directory:

```bash
PYTHONPATH=. python src/main.py
```

This script will:
1. Initialize the `DataManager` and download the specified dataset from Kaggle.
2. Initialize the `EvolutionEngine` with a population of models.
3. Run a continuous training loop for a specified number of generations.
4. Evaluate the population at regular intervals and evolve the best models.
5. Generate a sample text output from the final, best-performing model.

## Customization

You can customize the training process by modifying the hyperparameters in `src/main.py`:

- `population_size`: The number of models to train in parallel.
- `block_size`: The context length (sequence length) for the model.
- `batch_size`: The number of sequences per batch.
- `n_embd`: The embedding dimension.
- `n_head`: The number of attention heads.
- `n_layer`: The number of transformer blocks.
- `generations`: The number of evolutionary generations to run.
- `steps_per_generation`: The number of training steps before an evaluation/evolution phase.

To change the dataset, modify the `dataset_name` default parameter in the `DataManager` class in `src/data.py`. Ensure the dataset format is compatible (e.g., CSV with relevant text columns).
