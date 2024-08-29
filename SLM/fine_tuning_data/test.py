import pandas as pd
import numpy as np
import torch
from transformers import DistilBERTTokenizer, DistilBERTForQuestionAnswering
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

# Load the preprocessed text data and associated labels
train_data = pd.read_csv('train.csv')
val_data = pd.read_csv('val.csv')
test_data = pd.read_csv('test.csv')

# Define a custom dataset class
class PDFDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]
        labels = self.data.iloc[idx, 1]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

# Create a custom data loader
train_dataset = PDFDataset(train_data, DistilBERTTokenizer.from_pretrained('distilbert-base-uncased'))
val_dataset = PDFDataset(val_data, DistilBERTTokenizer.from_pretrained('distilbert-base-uncased'))
test_dataset = PDFDataset(test_data, DistilBERTTokenizer.from_pretrained('distilbert-base-uncased'))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define a custom model that inherits from the pre-trained DistilBERT model
class PDFQuestionAnsweringModel(DistilBERTForQuestionAnswering):
    def __init__(self, num_labels):
        super().__init__(num_labels)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = super().forward(input_ids, attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        outputs = torch.nn.CrossEntropyLoss()(pooled_output, torch.tensor([1]))
        return outputs

# Fine-tune the model
model = PDFQuestionAnsweringModel(num_labels=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        evaluation_strategy='epoch',
        learning_rate=1e-5,
        save_total_limit=2,
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        save_strategy='steps'
    ),
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=lambda pred: {'accuracy': accuracy_score(pred.label_ids, pred.predictions.argmax(-1))}
)

trainer.train()

# Evaluate the model on the testing set
test_results = trainer.evaluate(test_loader)
print(f'Test accuracy: {test_results.metrics["accuracy"]:.4f}')