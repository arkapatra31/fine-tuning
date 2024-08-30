from SLM.fine_tuning_data.dataExtractorAndProcessor.data_preprocessor import train_tokenized_datasets, \
    test_tokenized_datasets
# from SLM.models.DistilBERT.base_model import tokenizer
from SLM.fine_tuning_data.trainer.create_trainer_configuration import training_config
from transformers import Trainer, DataCollatorWithPadding, DistilBertForSequenceClassification, DistilBertTokenizerFast

# Initialize the model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Initialize the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


for elem in train_tokenized_datasets['name']:
    print(elem, type(elem))


# Define a custom data collator
class CustomDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # Check if features is a list and convert it to a dictionary
        if isinstance(features, list):
            features = {key: [feature[key] for feature in features] for key in features[0]}
        return super().__call__(features)


# Initialize the data collator
data_collator = CustomDataCollator(tokenizer)

# Ensure the datasets contain the necessary labels
# Assuming 'labels' is the key for the labels in the datasets
if 'labels' not in train_tokenized_datasets.column_names:
    train_tokenized_datasets = train_tokenized_datasets.map(lambda examples: {'labels': examples['label']})
if 'labels' not in test_tokenized_datasets.column_names:
    test_tokenized_datasets = test_tokenized_datasets.map(lambda examples: {'labels': examples['label']})


# Train the model
trained_model = Trainer(
    model=model,
    args=training_config,
    train_dataset=train_tokenized_datasets,
    eval_dataset=test_tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer
)

trained_model.train()

evaluation_result = trained_model.evaluate()
print(evaluation_result)

model.save_pretrained("./fine-tuned/")
tokenizer.save_pretrained("./fine-tuned")
