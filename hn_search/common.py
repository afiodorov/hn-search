"""Query embedding via ONNX Runtime (serve path — no torch/sentence-transformers).

The corpus is embedded offline with sentence-transformers/all-mpnet-base-v2 (see
misc/generate_embeddings_gpu.py). At serve time we only need to embed the user's
*query*, so we run the same model through ONNX Runtime instead of PyTorch:

  * ~300-450 MB resident instead of ~1.5 GB for torch + the model
  * sub-second cold start, no torch dependency tree
  * verified bit-exact (cosine 1.0) against sentence-transformers and against the
    stored corpus embeddings, so queries align with the indexed vectors

The ONNX weights and tokenizer are pulled from the model's Hugging Face repo
(which ships pre-exported ONNX), cached under HF_HOME. Pick the weight file via
HN_ONNX_MODEL_FILE — defaults to fp32 (exact); on ARM hosts (Oracle/Hetzner)
"onnx/model_qint8_arm64.onnx" is smaller/faster with negligible drift.
"""

import os

import numpy as np

from hn_search.logging_config import get_logger

MODEL_REPO = os.getenv("HN_MODEL_REPO", "sentence-transformers/all-mpnet-base-v2")
# fp32 = exact parity. For ARM serve use "onnx/model_qint8_arm64.onnx";
# for x86 a smaller option is "onnx/model_quint8_avx2.onnx".
ONNX_MODEL_FILE = os.getenv("HN_ONNX_MODEL_FILE", "onnx/model.onnx")
MAX_SEQ_LENGTH = int(os.getenv("HN_EMBED_MAX_LEN", "384"))

logger = get_logger(__name__)

# Singleton encoder (loaded once, reused across requests)
_encoder = None


class OnnxEncoder:
    """Drop-in replacement for the sentence-transformers encoder used at serve time.

    Exposes ``encode(list[str]) -> np.ndarray[N, 768]`` with mean pooling + L2
    normalization, matching all-mpnet-base-v2's pipeline.
    """

    def __init__(self):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        logger.info(f"🔧 Loading ONNX encoder ({MODEL_REPO} :: {ONNX_MODEL_FILE})...")
        model_path = hf_hub_download(MODEL_REPO, ONNX_MODEL_FILE)
        tokenizer_path = hf_hub_download(MODEL_REPO, "tokenizer.json")

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        pad_id = self.tokenizer.token_to_id("<pad>")
        self.tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
        self.tokenizer.enable_padding(
            pad_id=1 if pad_id is None else pad_id, pad_token="<pad>"
        )

        opts = ort.SessionOptions()
        threads = int(os.getenv("ORT_NUM_THREADS", "0"))
        if threads > 0:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            model_path, opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        logger.info(f"✅ ONNX encoder loaded (inputs: {sorted(self.input_names)})")

    def encode(self, texts, **_kwargs) -> np.ndarray:
        """Encode a list of strings (or a single string) to normalized embeddings.

        Extra kwargs (e.g. batch_size, convert_to_numpy) are accepted and ignored
        for compatibility with the sentence-transformers call sites.
        """
        if isinstance(texts, str):
            texts = [texts]

        encodings = self.tokenizer.encode_batch(list(texts))
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)
        feed = {k: v for k, v in feed.items() if k in self.input_names}

        token_embeddings = self.session.run(None, feed)[0]  # [B, T, 768]

        # Masked mean pooling over tokens, then L2 normalize (cosine-ready).
        mask = attention_mask[:, :, None].astype(np.float32)
        summed = (token_embeddings * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        embeddings = summed / counts
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings.astype(np.float32)


def get_model() -> OnnxEncoder:
    """Get or create the singleton ONNX query encoder."""
    global _encoder
    if _encoder is None:
        _encoder = OnnxEncoder()
    return _encoder
