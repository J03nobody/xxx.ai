import torch
import copy
from src.model import GPTLanguageModel

class EvolutionEngine:
    def __init__(self, population_size=2, vocab_size=50257, block_size=64, n_embd=64, n_head=4, n_layer=4, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.population_size = population_size
        self.device = device
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.vocab_size = vocab_size

        self.population = []
        self.optimizers = []

        for _ in range(population_size):
            model = GPTLanguageModel(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer)
            model.to(device)
            self.population.append(model)
            self.optimizers.append(torch.optim.AdamW(model.parameters(), lr=3e-4))

    def train_step(self, x, y, model_idx):
        model = self.population[model_idx]
        optimizer = self.optimizers[model_idx]

        model.train()
        x, y = x.to(self.device), y.to(self.device)

        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        return loss.item()

    def evaluate(self, model_idx, x_val, y_val):
        model = self.population[model_idx]
        model.eval()
        x_val, y_val = x_val.to(self.device), y_val.to(self.device)
        with torch.no_grad():
            _, loss = model(x_val, y_val)
        return loss.item()

    def evolve(self, val_losses):
        best_idx = val_losses.index(min(val_losses))
        best_loss = val_losses[best_idx]

        print(f"Evolution step: Best model index {best_idx} with loss {best_loss:.4f}")

        best_model = self.population[best_idx]
        best_state = copy.deepcopy(best_model.state_dict())

        new_population = []
        new_optimizers = []

        for i in range(self.population_size):
            new_model = GPTLanguageModel(vocab_size=self.vocab_size, block_size=self.block_size, n_embd=self.n_embd, n_head=self.n_head, n_layer=self.n_layer)
            new_model.load_state_dict(best_state)

            # Apply slight mutation to the weights to encourage diversity (simple evolutionary strategy)
            # Only mutate if it's not the first one (keep the best one exactly as is)
            if i > 0:
                with torch.no_grad():
                    for param in new_model.parameters():
                        noise = torch.randn_like(param) * 0.001  # Small Gaussian noise
                        param.add_(noise)

            new_model.to(self.device)
            new_population.append(new_model)

            new_optimizers.append(torch.optim.AdamW(new_model.parameters(), lr=3e-4))

        self.population = new_population
        self.optimizers = new_optimizers

        return best_idx
