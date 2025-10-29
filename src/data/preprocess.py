import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split

# Load raw data
df = pd.read_csv("data/raw/tickets_raw.csv")

# --- Basic cleaning ---
df = df.dropna(subset=["text", "category"])        # drop empty rows
df["text"] = df["text"].str.strip()                # clean spaces
df["category"] = df["category"].str.lower().str.strip()  # normalize labels

# --- Encode labels ---
labels = sorted(df["category"].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}
df["label"] = df["category"].map(label2id)

# --- Split into train/val ---
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

# --- Save processed files ---
os.makedirs("data/processed", exist_ok=True)
train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/val.csv", index=False)

# --- Save label mapping ---
with open("data/processed/label2name.json", "w") as f:
    json.dump(id2label, f, indent=2)

print("✅ Preprocessing complete.")
print(f"Classes: {len(labels)} | Saved to data/processed/")
