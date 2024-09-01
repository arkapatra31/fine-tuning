import torch
from transformers import DistilBertModel, DistilBertTokenizerFast

# Load pre-trained model tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')


__all__ = [
    model, tokenizer
]
