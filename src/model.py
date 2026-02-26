import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32
block_size = 64
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2
vocab_size = 50304 # GPT-2 vocab size roughly

class Head(nn.Module):
""" one head of self-attention """

def __init__(self, head_size, n_embd, block_size):
super().__init__()
self.key = nn.Linear(n_embd, head_size, bias=False)
self.query = nn.Linear(n_embd, head_size, bias=False)
self.value = nn.Linear(n_embd, head_size, bias=False)
self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

self.dropout = nn.Dropout(dropout)

def forward(self, x):
# input of size (batch, time-step, channels)
# output of size (batch, time-step, head size)
B,T,C = x.shape
k = self.key(x)   # (B,T,hs)
q = self.query(x) # (B,T,hs)
# compute attention scores ("affinities")
wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
wei = F.softmax(wei, dim=-1) # (B, T, T)
wei = self.dropout(wei)
# perform the weighted aggregation of the values
v = self.value(x) # (B,T,hs)
out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
return out

class MultiHeadAttention(nn.Module):
""" multiple heads of self-attention in parallel """

def __init__(self, num_heads, head_size, n_embd, block_size):
super().__init__()
self.heads = nn.ModuleList([Head(head_size, n_embd, block_size) for _ in range(num_heads)])
self.proj = nn.Linear(head_size * num_heads, n_embd)
self.dropout = nn.Dropout(dropout)

def forward(self, x):
out = torch.cat([h(x) for h in self.heads], dim=-1)
out = self.proj(out)
out = self.dropout(out)
return out

class FeedFoward(nn.Module):
""" a simple linear layer followed by a non-linearity """

def __init__(self, n_embd):
super().__init__()
self.net = nn.Sequential(
	nn.Linear(n_embd, 4 * n_embd),
	nn.ReLU(),
	nn.Linear(4 * n_embd, n_embd),
	nn.Dropout(dropout),
	)
	
	def forward(self, x):
	return self.net(x)
	
	class Block(nn.Module):
	""" Transformer block: communication followed by computation """
	
	def __init__(self, n_embd, n_head, block_size):
	# n_embd: embedding dimension, n_head: the number of heads we'd like
	super().__init__()
	head_size = n_embd // n_head
	self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size)
	self.ffwd = FeedFoward(n_embd)
	self.ln1 = nn.LayerNorm(n_embd)
	self.ln2 = nn.LayerNorm(n_embd)
	
	def forward(self, x):
	x = x + self.sa(self.ln1(x))
	x = x + self.ffwd(self.ln2(x))
	return x
	
	class GPTLanguageModel(nn.Module):
	
	def __init__(self, vocab_size=vocab_size, block_size=block_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer):
	super().__init__()
	self.block_size = block_size
	# each token directly reads off the logits for the next token from a lookup table
	self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
	self.position_embedding_table = nn.Embedding(block_size, n_embd)
	self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, block_size=block_size) for _ in range(n_layer)])
	self.ln_f = nn.LayerNorm(n_embd) # final layer norm
	self.lm_head = nn.Linear(n_embd, vocab_size)
	
	# better init
	self.apply(self._init_weights)
	
	def _init_weights(self, module):
	if isinstance(module, nn.Linear):
		torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
		if module.bias is not None:
			torch.nn.init.zeros_(module.bias)
			elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
			
			def forward(self, idx, targets=None):
			B, T = idx.shape
			
			# idx and targets are both (B,T) tensor of integers
			tok_emb = self.token_embedding_table(idx) # (B,T,C)
			pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
			x = tok_emb + pos_emb # (B,T,C)
			
			# Apply blocks
			for block in self.blocks:
				x = block(x)
				
				x = self.ln_f(x) # (B,T,C)
				logits = self.lm_head(x) # (B,T,vocab_size)
				
				if targets is None:
					loss = None
					else:
						B, T, C = logits.shape
						logits = logits.view(B*T, C)
						targets = targets.view(B*T)
						loss = F.cross_entropy(logits, targets)
						
						return logits, loss
						
						def generate(self, idx, max_new_tokens):
						# idx is (B, T) array of indices in the current context
						for _ in range(max_new_tokens):
							# crop idx to the last block_size tokens
							idx_cond = idx[:, -self.block_size:]
							# get the predictions
							logits, _ = self(idx_cond)
							# focus only on the last time step
							logits = logits[:, -1, :] # becomes (B, C)
							# apply softmax to get probabilities
							probs = F.softmax(logits, dim=-1) # (B, C)
							# sample from the distribution
							idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
							# append sampled index to the running sequence
							idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
							return idx
							EOF
							
							# Create src/data.py
							cat <<EOF > src/data.py
							import torch
							from datasets import load_dataset
							from transformers import AutoTokenizer
							
							class DataManager:
							def __init__(self, dataset_name="wikitext", config_name="wikitext-2-raw-v1", split="train", tokenizer_name="gpt2", batch_size=32, block_size=64):
							self.dataset = load_dataset(dataset_name, config_name, split=split, streaming=True)
							self.dataset_iterator = iter(self.dataset)
							self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
							if self.tokenizer.pad_token is None:
								self.tokenizer.pad_token = self.tokenizer.eos_token
								self.batch_size = batch_size
								self.block_size = block_size
								
								def get_batch(self):
								data = []
								
								while len(data) < self.batch_size:
									try:
									item = next(self.dataset_iterator)
									except StopIteration:
									self.dataset_iterator = iter(self.dataset)
									item = next(self.dataset_iterator)
									
									text = item['text']
									encoded = self.tokenizer.encode(text)
									
									if len(encoded) > self.block_size:
										tensor_encoded = torch.tensor(encoded, dtype=torch.long)
										data.append(tensor_encoded)
										
										x_batch = []
										y_batch = []
										
										for d in data:
											if len(x_batch) == self.batch_size:
												break
												
												ix = torch.randint(len(d) - self.block_size, (1,)).item()
												x = d[ix : ix + self.block_size]
												y = d[ix + 1 : ix + self.block_size + 1]
												x_batch.append(x)
												y_batch.append(y)
												
												x_batch = torch.stack(x_batch)
												y_batch = torch.stack(y_batch)
												
												return x_batch, y_batch
												
												def get_vocab_size(self):
												return self.tokenizer.vocab_size
												
												if __name__ == "__main__":
													dm = DataManager()
													x, y = dm.get_batch()
													print("X shape:", x.shape)
													print("Y shape:", y.shape)
													print("Vocab size:", dm.get_vocab_size())
													EOF
													
													# Create src/engine.py
													cat <<EOF > src/engine.py
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
																	EOF
																	
																	# Create src/main.py
																	cat <<EOF > src/main.py
																	import torch
																	import time
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
						EOF
						
						# Create .gitignore
						cat <<EOF > .gitignore
						__pycache__/
						*.pyc
						.ipynb_checkpoints/
						.env
						data/
						EOF
						
						echo "XXX.AI project setup complete!"
						
