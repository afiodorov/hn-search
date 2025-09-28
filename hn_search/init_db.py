import json
from pathlib import Path

import html2text

from hn_search.common import COLLECTION_NAME, get_client, get_device, get_model


def strip_html(text: str) -> str:
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.ignore_emphasis = True
    return h.handle(text).strip()


def init_db(
    data_path: str = "data.jsonl",
    chroma_host: str = "localhost",
    chroma_port: int = 8000,
):
    client = get_client(host=chroma_host, port=chroma_port)

    device = get_device()
    print(f"Using device: {device}")
    model = get_model(device=device)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    documents = []
    metadatas = []
    ids = []

    with open(data_file) as f:
        for line in f:
            item = json.loads(line)
            clean_text = strip_html(item["text"])
            documents.append(clean_text)
            metadatas.append(
                {
                    "author": item["author"],
                    "timestamp": item["timestamp"],
                    "type": item["type"],
                }
            )
            ids.append(item["id"])

    print(f"Loading {len(documents)} documents into ChromaDB...")

    batch_size = 5_000
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]

        embeddings = model.encode(batch_docs, show_progress_bar=True)

        collection.add(
            documents=batch_docs,
            embeddings=embeddings.tolist(),
            metadatas=batch_meta,
            ids=batch_ids,
        )
        print(
            f"Processed {min(i + batch_size, len(documents))}/{len(documents)} documents"
        )

    print(f"Successfully loaded {len(documents)} documents into ChromaDB")
    print(f"Collection size: {collection.count()}")


if __name__ == "__main__":
    init_db()
