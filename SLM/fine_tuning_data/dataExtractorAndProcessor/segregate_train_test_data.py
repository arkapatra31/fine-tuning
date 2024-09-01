from SLM.fine_tuning_data.dataExtractorAndProcessor.AmazonReviewDataset import AmazonReviewsDataset
from sklearn.model_selection import train_test_split
from SLM.fine_tuning_data.dataExtractorAndProcessor.data_sanitization import df
from transformers import DistilBertTokenizerFast

"""
Documenting the train_test_split function below :- 
Split arrays or matrices into random train and test subsets.
Quick utility that wraps input validation, ``next(ShuffleSplit().split(X, y))``, and application to input data into a 
single call for splitting (and optionally subsampling) data into a one-liner.
"""

# Initialise the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Data preparation
MAX_LEN = 128
train_texts, val_texts, train_labels, val_labels = train_test_split(df['review'], df['rating'], test_size=0.1)

train_dataset = AmazonReviewsDataset(train_texts.tolist(), train_labels.tolist(), tokenizer, MAX_LEN)
val_dataset = AmazonReviewsDataset(val_texts.tolist(), val_labels.tolist(), tokenizer, MAX_LEN)

__all__ = [
    train_dataset, val_dataset
]