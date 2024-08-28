import torch
from transformers import AutoModel, AutoTokenizer


device = "cuda"

# Load pre-trained model tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModel.from_pretrained('distilbert-base-uncased')

# Save the model locally
model.save_pretrained("./pretrained./")
tokenizer.save_pretrained("./pretrained")
