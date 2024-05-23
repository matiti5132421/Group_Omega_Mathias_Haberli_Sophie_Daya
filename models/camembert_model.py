
import pandas as pd
from transformers import CamembertTokenizer, CamembertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_metric
from sklearn.metrics import classification_report
import numpy as np

def camembert_model(training_data_path):
    # Load the training data
    training_data_pd = pd.read_csv(training_data_path)

    # Sample a smaller dataset for testing purposes
    training_data_pd = training_data_pd.sample(frac=0.05, random_state=42)

    # Load the tokenizer
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')

    # Function to encode data
    def encode_data(tokenizer, df):
        texts = df['sentence'].tolist()
        labels = df['difficulty'].map({'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5}).tolist()
        encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=128)
        return Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        })

    # Split the DataFrame into training and validation sets
    train_data, val_data = train_test_split(training_data_pd, test_size=0.1, random_state=42)

    # Tokenize and prepare datasets
    train_dataset = encode_data(tokenizer, train_data)
    val_dataset = encode_data(tokenizer, val_data)

    # Load the model
    model = CamembertForSequenceClassification.from_pretrained('camembert-base', num_labels=6)

    # Metric for evaluation
    def compute_metrics(eval_pred):
        metric = load_metric("accuracy")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = metric.compute(predictions=predictions, references=labels)
        return {"accuracy": accuracy['accuracy']}

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,  # Reduced number of epochs
        per_device_train_batch_size=2,  # Further reduce the batch size
        gradient_accumulation_steps=2,  # Increase gradient accumulation steps
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        learning_rate=2e-5,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    predictions, labels, _ = trainer.predict(val_dataset)
    predictions = np.argmax(predictions, axis=-1)

    # Generate classification report
    report = classification_report(labels, predictions, target_names=['A1', 'A2', 'B1', 'B2', 'C1', 'C2'], output_dict=True)

    # Extract detailed performance metrics and display class-wise results
    class_results_df = pd.DataFrame(report).transpose().iloc[:-3, :-1]

    # Creating a DataFrame to display overall results
    overall_results_df = pd.DataFrame({
        "Precision": [report['weighted avg']['precision']],
        "Recall": [report['weighted avg']['recall']],
        "F1-Score": [report['weighted avg']['f1-score']],
        "Accuracy": [report['accuracy']]
    }, index=["Camembert"])

    # Displaying the DataFrames
    print("Camembert Model Evaluation (Class-wise)")
    print(class_results_df)

    print("\nCamembert Model Evaluation (Overall)")
    print(overall_results_df)

    return model, tokenizer

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate a Camembert model.")
    parser.add_argument('training_data_path', type=str, help="Path to the training data CSV file.")

    args = parser.parse_args()

    model, tokenizer = camembert_model(args.training_data_path)

    print("Model training and evaluation complete.")
