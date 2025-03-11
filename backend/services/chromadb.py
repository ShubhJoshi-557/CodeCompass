import chromadb

# Initialize ChromaDB Client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("code_embeddings", metadata={"hnsw:batch_size":1000000})
