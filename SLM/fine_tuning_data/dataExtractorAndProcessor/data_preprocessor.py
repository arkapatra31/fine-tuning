import os
from SLM.fine_tuning_data.dataExtractorAndProcessor.segregate_train_test_data import train_dataset, test_dataset
from dotenv import load_dotenv
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments, DistilBertTokenizerFast
from datasets import load_dataset, DatasetDict

load_dotenv()

base_slm = os.getenv("BASE_SLM")

# Initialize the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


# Preprocess the data
def preprocess_function(examples):
    # Combine relevant columns into a single text field
    text = [f"{name} {date} {rating} {review}" for name, date, rating, review in
            zip(examples['name'], examples['date'], examples['rating'], examples['review'])]
    return tokenizer(text, padding="max_length", truncation=True)


# Apply the preprocessing function to the dataset
train_tokenized_datasets = train_dataset.map(preprocess_function, batched=True)
test_tokenized_datasets = test_dataset.map(preprocess_function, batched=True)

__all__ = [
    train_tokenized_datasets,
    test_tokenized_datasets
]
