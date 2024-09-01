from datasets import load_dataset
import pandas as pd

# Load dataset
dataset = load_dataset("arkapatra31/datasets")["train"]
df = dataset.to_pandas()

# Also the data can be read locally from a csv
#df = pd.read_csv("product_reviews.csv")

print(df.head())
__all__ = [
    df
]
