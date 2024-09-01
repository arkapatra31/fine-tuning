import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Load the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Load the pre-trained DistilBERT model
pretrained_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=5)

# Load the fine-tuned DistilBERT model
fine_tuned_model = DistilBertForSequenceClassification.from_pretrained('fine_tuned_distilbert')

# Example input text
input_text = "The product is good although it seems to be a little overpriced"

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding=True, max_length=128)

# Get outputs from the pre-trained model
pretrained_model.eval()
with torch.no_grad():
    pretrained_outputs = pretrained_model(**inputs)
    pretrained_logits = pretrained_outputs.logits
    pretrained_prediction = torch.argmax(pretrained_logits, dim=1).item()

# Get outputs from the fine-tuned model
fine_tuned_model.eval()
with torch.no_grad():
    fine_tuned_outputs = fine_tuned_model(**inputs)
    fine_tuned_logits = fine_tuned_outputs.logits
    fine_tuned_prediction = torch.argmax(fine_tuned_logits, dim=1).item()

# Print the predictions
print(f"Pre-trained Model Prediction: {pretrained_prediction + 1}")  # Add 1 to convert back to 1-5 scale
print(f"Fine-tuned Model Prediction: {fine_tuned_prediction + 1}")  # Add 1 to convert back to 1-5 scale

# Print the logits for more detailed comparison
print(f"Pre-trained Model Logits: {pretrained_logits}")
print(f"Fine-tuned Model Logits: {fine_tuned_logits}")
