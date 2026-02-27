import sys
import os
import torch

# Add current directory to path
sys.path.append(os.getcwd())

def verify():
    print("Verifying rescue...")
    try:
        from src.model import GPTLanguageModel, GPTConfig
        print("Successfully imported src.model")
    except Exception as e:
        print(f"Failed to import src.model: {e}")
        return False

    try:
        from src.data import DataManager
        print("Successfully imported src.data")
    except Exception as e:
        print(f"Failed to import src.data: {e}")
        return False

    try:
        from src.engine import EvolutionEngine
        print("Successfully imported src.engine")
    except Exception as e:
        print(f"Failed to import src.engine: {e}")
        return False

    try:
        from src.main import main
        print("Successfully imported src.main")
    except Exception as e:
        print(f"Failed to import src.main: {e}")
        return False

    print("All imports successful!")

    # Quick instantiation check
    try:
        config = GPTConfig()
        model = GPTLanguageModel(config)
        print("Model instantiated successfully")
    except Exception as e:
        print(f"Failed to instantiate model: {e}")
        return False

    return True

if __name__ == "__main__":
    if verify():
        sys.exit(0)
    else:
        sys.exit(1)
