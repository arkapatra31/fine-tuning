from transformers import TrainingArguments

#Define training arguments
training_config = TrainingArguments(
    output_dir="./results",
    eval_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
)

__all__ = [
    training_config
]