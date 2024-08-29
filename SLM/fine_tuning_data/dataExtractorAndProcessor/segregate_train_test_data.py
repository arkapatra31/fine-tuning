from SLM.fine_tuning_data.dataExtractorAndProcessor.load_kaggle_dataset import dataset

# Split the train and the test data
dataset = dataset['train'].train_test_split(test_size=0.1)
print(dataset)
train_dataset = dataset['train']
test_dataset = dataset['test']

__all__ = [
    train_dataset, test_dataset
]
