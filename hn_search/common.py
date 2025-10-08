import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Singleton embedding model
_model = None


def get_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def get_model(device=None):
    """Get or create singleton embedding model."""
    global _model
    if _model is None:
        if device is None:
            device = get_device()
        print(f"🔧 Loading embedding model on {device}...")
        _model = SentenceTransformer(MODEL_NAME, device=device)
        print(f"✅ Embedding model loaded")
    return _model
