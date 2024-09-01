from SLM.fine_tuning_data.dataExtractorAndProcessor.segregate_train_test_data import train_dataset, val_dataset
from transformers import Trainer, DataCollatorWithPadding, DistilBertForSequenceClassification, DistilBertTokenizerFast, \
    DistilBertTokenizer
from SLM.trainer.create_trainer_configuration import training_args
from SLM.models.DistilBERT.fine_tuning_model.configure_training_device import device
import torch

# Initialise the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Initialise the model.
"""
Also note that num_labels can change, here I'm assuming the model needs to be fine-tuned in order to predict that many
number of labels or classes
"""
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=5)

# Move model to device which is CPU in this instance
model.to(device)

# Initialise the DistilBERT trainer with the necessary defined config
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# Fine-tuning the model
trainer.train()

# Now we can save the fine-tuned model to specified directory which can be used for later purpose
# Save the fine-tuned model
model.save_pretrained("fine-tuned-distilbert")
tokenizer.save_pretrained("fine-tuned-distilbert")