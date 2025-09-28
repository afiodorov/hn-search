import sys

from hn_search.common import COLLECTION_NAME, get_client, get_device, get_model


def query(
    query_text: str,
    n_results: int = 10,
    chroma_host: str = "localhost",
    chroma_port: int = 8000,
):
    client = get_client(host=chroma_host, port=chroma_port)

    device = get_device()
    print(f"Loading embedding model on {device}...", file=sys.stderr)
    model = get_model(device=device)

    print("Encoding query...", file=sys.stderr)
    query_embedding = model.encode([query_text])

    print("Querying ChromaDB...", file=sys.stderr)
    collection = client.get_collection(name=COLLECTION_NAME)

    results = collection.query(
        query_embeddings=query_embedding.tolist(), n_results=n_results
    )

    print("Fetching results...\n", file=sys.stderr)
    sys.stdout.flush()

    for i, (doc_id, document, metadata, distance) in enumerate(
        zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        1,
    ):
        print(f"=== Result {i} (distance: {distance:.4f}) ===")
        print(f"ID: {doc_id}")
        print(f"Author: {metadata['author']}")
        print(f"Timestamp: {metadata['timestamp']}")
        print(f"Text: {document}")
        print()
        sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m hn_search.query <query_text> [n_results]")
        sys.exit(1)

    query_text = sys.argv[1]
    n_results = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    query(query_text, n_results)
