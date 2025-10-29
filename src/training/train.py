# src/training/train.py
import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import mlflow

MODEL_NAME = os.getenv("PRETRAINED", "distilbert-base-uncased")
OUTPUT_DIR = "models/model-1"

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # expected columns: subject, body, label
    df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
    return Dataset.from_pandas(df[["text","label"]])

def main():
    mlflow.set_experiment("ticket-classifier")
    with mlflow.start_run():
        train_ds = load_data("data/processed/train.csv")
        val_ds = load_data("data/processed/val.csv")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        def tokenize(batch):
            return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)
        train_ds = train_ds.map(tokenize, batched=True)
        val_ds = val_ds.map(tokenize, batched=True)
        train_ds.set_format(type="torch", columns=["input_ids","attention_mask","label"])
        val_ds.set_format(type="torch", columns=["input_ids","attention_mask","label"])

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            num_train_epochs=3,
            logging_dir="./logs",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        metrics = trainer.evaluate()
        # Save artifacts and log with MLflow
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        mlflow.log_params({
            "model_name": MODEL_NAME,
            "epochs": 3
        })
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(OUTPUT_DIR + "/pytorch_model.bin")
        # register model in MLflow if desired
        print("Training finished.")

if __name__ == "__main__":
    NUM_LABELS = 4  # set according to your dataset
    main()
