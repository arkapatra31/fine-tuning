from torch.utils.data import Dataset
from SLM.models.DistilBERT.fine_tuning_model.configure_training_device import device
import torch


# Create the abstract class for the Amazon reviews extending the Datase class
class AmazonReviewsDataset(Dataset):
    def __init__(self, review, labels, tokenizer, max_len):
        self.review = review
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.review)

    def __getitem__(self, index):
        review = self.review[index]
        label = self.labels[index]
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'review': review,
            'input_ids': encoding['input_ids'].flatten().to(device),
            'attention_mask': encoding['attention_mask'].flatten().to(device),
            'labels': torch.tensor(label, dtype=torch.long).to(device)
        }


__all__ = [
    AmazonReviewsDataset
]
