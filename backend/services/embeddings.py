import chromadb
import asyncio
import json
import logging
import traceback
from sentence_transformers import SentenceTransformer
from typing import List
from services.chromadb import collection  # Import collection from db.py

# Enable Debug Logging for ChromaDB
logging.basicConfig(level=logging.DEBUG)

# Load Embedding Model (Singleton)
embedding_model = SentenceTransformer("BAAI/bge-small-en")

def chunk_text(text: str, chunk_size: int = 256) -> List[str]:
    """Splits text into smaller overlapping chunks for better retrieval."""
    words = text.split()
    return [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size)]

def generate_embeddings(text_chunks: List[str]) -> List[List[float]]:
    """Generates vector embeddings for text chunks."""
    return embedding_model.encode(text_chunks, convert_to_numpy=True).tolist()

BATCH_SIZE = 500  # Reduce if still failing

async def store_embeddings(owner: str, repo: str, file_path: str, text: str):
    try:
        print(f"🟡 Processing file: {file_path}")

        chunks = chunk_text(text)
        if not chunks:
            print(f"⚠️ No chunks generated for {file_path}")
            return

        print(f"📌 Chunks generated: {len(chunks)}")

        vectors = generate_embeddings(chunks)
        if not vectors or len(vectors) != len(chunks):
            print(f"⚠️ Embedding mismatch! Chunks: {len(chunks)}, Vectors: {len(vectors) if vectors else 'None'}")
            return

        print(f"📌 Embeddings generated: {len(vectors)}")

        documents = [{"owner": owner, "repo": repo, "file_path": file_path, "chunk_index": idx} for idx in range(len(chunks))]
        ids = [f"{owner}/{repo}/{file_path}::chunk{idx}" for idx in range(len(chunks))]

        print(f"📌 Storing {len(chunks)} chunks in ChromaDB in batches...")

        # Batch Processing
        for i in range(0, len(chunks), BATCH_SIZE):
            batch_ids = ids[i:i+BATCH_SIZE]
            batch_vectors = vectors[i:i+BATCH_SIZE]
            batch_documents = documents[i:i+BATCH_SIZE]
            batch_chunks = chunks[i:i+BATCH_SIZE]

            await asyncio.to_thread(
                collection.add,
                ids=batch_ids,
                embeddings=batch_vectors,
                metadatas=batch_documents,
                documents=batch_chunks
            )

            print(f"✅ Stored batch {i // BATCH_SIZE + 1} ({len(batch_ids)} items)")

        print(f"✅ Successfully stored {len(chunks)} chunks for {file_path}")

        existing = collection.get()
        print(f"🔥 ChromaDB currently holds {len(existing['ids'])} vectors!")
    except AssertionError as ae:
        print(f"❌ Data mismatch error: {ae}")
    except Exception as e:
        print(f"❌ Exception storing embeddings for {file_path}: {e}")
        traceback.print_exc()