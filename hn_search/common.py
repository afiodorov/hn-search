import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def get_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def get_model(device=None):
    if device is None:
        device = get_device()
    return SentenceTransformer(MODEL_NAME, device=device)
