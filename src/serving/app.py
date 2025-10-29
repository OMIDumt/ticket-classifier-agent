import os
import json
import requests
from fastapi import FastAPI, Request
from pydantic import BaseModel

# ------------------------
# CONFIG
# ------------------------
app = FastAPI()

HF_API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased"
HF_TOKEN = os.getenv("HF_TOKEN")  # must be set in Render environment variables
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

ESCALATE_THRESHOLD = float(os.getenv("ESCALATE_THRESHOLD", "0.80"))
CRITICAL_KEYWORDS = [
    "refund", "charge", "charged", "broken", "crash", "urgent", 
    "down", "error", "data loss", "fraud", "unauthorized"
]

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
async def predict(req: PredictRequest):
    """Main prediction endpoint"""
    text = req.text or (req.subject or "") + " " + (req.body or "")

    # Call Hugging Face API
    response = requests.post(HF_API_URL, headers=headers, json={"inputs": text})
    result = response.json()

    if "error" in result:
        return {"error": result["error"], "message": "Hugging Face inference failed"}

    # Parse result safely
    try:
        label = result[0][0]["label"]
        score = float(result[0][0]["score"])
    except Exception:
        label, score = "unknown", 0.0

    # Escalation logic
    lowered = text.lower()
    has_keyword = any(k in lowered for k in CRITICAL_KEYWORDS)
    escalate = bool(score >= ESCALATE_THRESHOLD or has_keyword)

    return {
        "id": req.id,
        "label": label,
        "confidence": score,
        "escalate": escalate
    }

@app.get("/")
def root():
    return {"status": "✅ Ticket Classifier API running on Render!"}
