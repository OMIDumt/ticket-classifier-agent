import os
import json
from fastapi import FastAPI, Request

from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# ------------------------
# CONFIG
# ------------------------
app = FastAPI(title="Ticket Classifier API")

# Use Hugging Face model if local model doesn't exist
MODEL_NAME = os.getenv("PRETRAINED", "distilbert-base-uncased-finetuned-sst-2-english")
MODEL_DIR = os.getenv("MODEL_DIR", "models/model-1")
ESCALATE_THRESHOLD = float(os.getenv("ESCALATE_THRESHOLD", "0.80"))
CRITICAL_KEYWORDS = ["refund", "charge", "charged", "broken", "crash", "urgent", "down", "error", "data loss", "fraud", "unauthorized"]

# ------------------------
# LOAD MODEL
# ------------------------
if os.path.exists(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, "config.json")):
    print(f"✅ Loading local fine-tuned model from {MODEL_DIR}")
    model_path = MODEL_DIR
else:
    print(f"🌐 Loading pretrained model from Hugging Face: {MODEL_NAME}")
    model_path = MODEL_NAME

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ------------------------
# LOAD LABEL MAP
# ------------------------
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label2name.json")
if os.path.exists(LABEL_MAP_PATH):
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        id2label = json.load(f)
        id2label = {int(k): v for k, v in id2label.items()}
else:
    id2label = {0: "other", 1: "Billing inquiry", 2: "Technical issue", 3: "Refund request"}

# ------------------------
# REQUEST MODEL
# ------------------------
class PredictRequest(BaseModel):
    text: str = ""
    id: str = None
    subject: str = None
    body: str = None

# ------------------------
# ENDPOINT
# ------------------------
@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    text = data["text"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return {"label": str(predicted_class)}

    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=256).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        label = id2label.get(idx, str(idx))
        confidence = float(probs[idx])

    # Escalation rule: high confidence or contains critical keywords
    lowered = text.lower()
    has_keyword = any(k in lowered for k in CRITICAL_KEYWORDS)
    escalate = bool(confidence >= ESCALATE_THRESHOLD or has_keyword)

    return {
        "id": payload.id,
        "label": label,
        "confidence": confidence,
        "escalate": escalate,
        "probs": probs.tolist()
    }
