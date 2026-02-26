import torch
from src.data import DataManager
from src.engine import EvolutionEngine

def main():
    # Configuration
    population_size = 2
    block_size = 64
    batch_size = 32
    n_embd = 64
    n_head = 4
    n_layer = 4

    # Simulate a "continuous" protocol by running for a fixed number of generations for this demo
    generations = 5
    steps_per_generation = 10

    print("Initializing Data Manager...")
    dm = DataManager(block_size=block_size, batch_size=batch_size)
    vocab_size = dm.get_vocab_size()

    print(f"Initializing Evolution Engine with population size {population_size}...")
    engine = EvolutionEngine(
        population_size=population_size,
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer
    )

    print("Starting continuous training protocol...")

    for gen in range(generations):
        print(f"\n--- Generation {gen+1}/{generations} ---")

        # Training phase
        # In a real scenario, we might download new data here
        print("Gathering data and training...")

        for step in range(steps_per_generation):
            x_batch, y_batch = dm.get_batch()

            step_losses = []
            for i in range(population_size):
                # We need to make sure models are on the same device and training independently
                # engine.train_step handles this
                loss = engine.train_step(x_batch, y_batch, i)
                step_losses.append(loss)

            if step % 5 == 0:
                print(f"  Step {step}/{steps_per_generation}: Losses {['%.4f' % l for l in step_losses]}")

        # Evaluation phase
        # Use a fresh batch as validation set (approximation)
        print("Evaluating population...")
        x_val, y_val = dm.get_batch()

        val_losses = []
        for i in range(population_size):
            val_loss = engine.evaluate(i, x_val, y_val)
            val_losses.append(val_loss)

        print(f"  Validation Losses: {['%.4f' % l for l in val_losses]}")

        # Evolution phase
        best_idx = engine.evolve(val_losses)
        print(f"  Winner of generation {gen+1}: Model {best_idx}")

    print("\nTraining complete.")

    # Demonstrate generation from the best model
    # Since all models are now clones of the best, we can pick index 0
    print("\nGenerating sample text from the final model...")
    model = engine.population[0]
    model.eval()

    # Start with a dummy token (e.g., 0) or a known start token.
    # GPT-2 tokenizer usually has specific start tokens, but here we can just pass a simple input.
    # Let's encode a prompt.
    prompt = "Once upon a time"
    # Ensure input_ids is on the correct device
    input_ids = torch.tensor(dm.tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(engine.device)

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=50)
        output_text = dm.tokenizer.decode(output_ids[0].tolist())

    print(f"Generated text:\n{output_text}")

if __name__ == "__main__":
    main()
