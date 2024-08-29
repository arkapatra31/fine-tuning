from datasets import load_dataset

# Load dataset from Kaggle using load_dataset only if the dataset exists in HGGF
dataset = load_dataset('arkapatra31/datasets')
__all__ = [
    dataset
]
