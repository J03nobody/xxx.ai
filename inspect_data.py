from datasets import load_dataset

try:
    print("Loading TinyStories dataset (train split, streaming)...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    print("Dataset loaded successfully.")

    print("Fetching first item...")
    item = next(iter(dataset))
    print("First item keys:", item.keys())
    print("First item content sample:", str(item)[:200])

except Exception as e:
    print(f"Error: {e}")
