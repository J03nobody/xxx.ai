import torch
import os
import kagglehub
from datasets import load_dataset
from transformers import AutoTokenizer

class DataManager:
    def __init__(self, dataset_name="wikitext", config_name="wikitext-2-raw-v1", split="train", tokenizer_name="gpt2", batch_size=32, block_size=64, kaggle_dataset=None, text_column="text"):
        if kaggle_dataset:
            path = kagglehub.dataset_download(kaggle_dataset)
            files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
            if not files:
                raise ValueError(f"No CSV files found in kaggle dataset {kaggle_dataset} at {path}")
            self.dataset = load_dataset('csv', data_files=files, split=split, streaming=True)
        else:
            self.dataset = load_dataset(dataset_name, config_name, split=split, streaming=True)
        self.text_column = text_column
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

            text = item.get(self.text_column, "")
            if not text or not isinstance(text, str):
                continue
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
