from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from concurrent.futures import as_completed
import faiss
import numpy as np
import requests
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from celery import Celery
import git
import os
import redis
import uuid
import pickle
import time
import torch

# Database Setup
DATABASE_URL = "sqlite:///./codecompass.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# FAISS Setup (Vector Search)
d = 384  # Dimension of embeddings
faiss_index_path = "faiss.index"
cached_embeddings_path = "embeddings.pkl"

# Load or Initialize FAISS index
if os.path.exists(faiss_index_path):
    index = faiss.read_index(faiss_index_path)
else:
    index = faiss.IndexFlatL2(d)

# Load or Initialize Embedding Cache
if os.path.exists(cached_embeddings_path):
    with open(cached_embeddings_path, "rb") as f:
        embedding_cache = pickle.load(f)
else:
    embedding_cache = {}

BATCH_SIZE = 512  # Number of files to process in each batch
MODEL_BATCH_SIZE = 64  # Batch size for transformer model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU acceleration
MAX_CHUNK_LINES = 400  # Lines per chunk
OVERLAP_LINES = 20     # Overlap between chunks

# Load Sentence Transformer Model with GPU support
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

# Precompiled filter sets
VALID_EXTENSIONS = frozenset({
    ".py", ".js", ".jsx", ".ts", ".tsx", ".cpp", ".c", ".cc", ".h", ".hpp",
    ".java", ".cs", ".go", ".rs", ".swift", ".kt", ".kts", ".dart", ".rb",
    ".php", ".pl", ".pm", ".lua", ".r", ".sh", ".bash", ".zsh", ".fish",
    ".bat", ".cmd", ".ps1", ".vue", ".svelte", ".pug", ".jade", ".tf", "Dockerfile"
})

BLACKLIST_FOLDERS = frozenset({
    "node_modules", "vendor", "bin", "obj", "dist", "build", "__pycache__",
    ".git", ".svn", ".hg", ".vscode", ".idea", ".vs", "target", "out", 
    "tmp", "cache", "logs", "__snapshots__", "__tests__", "test", "examples"
})

BLACKLIST_FILES = frozenset({
    "package-lock.json", "yarn.lock", "Makefile", "README.md", "LICENSE",
    "CHANGELOG.md", ".DS_Store", "Thumbs.db", ".npmrc", ".yarnrc", "go.mod",
    "go.sum", "Cargo.lock", "Pipfile.lock", "poetry.lock"
})

BLACKLIST_EXTENSIONS = frozenset({
    ".json", ".xml", ".yaml", ".yml", ".lock", ".md", ".markdown", ".txt",
    ".ini", ".log", ".tmp", ".iml", ".sln", ".csproj", ".classpath",
    ".project", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".mp3",
    ".mp4", ".avi", ".pdf", ".docx", ".xlsx", ".zip", ".tar", ".gz", 
    ".7z", ".rar", ".bin", ".exe", ".dll", ".dylib", ".so"
})

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Redis Setup for Task Tracking & Caching
redis_client = redis.Redis(host="127.0.0.1", port=6379, db=0)

# Celery Configuration
celery = Celery("tasks", broker="redis://127.0.0.1:6379/0", backend="redis://127.0.0.1:6379/0")

# Models
class Repository(Base):
    __tablename__ = "repositories"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    owner = Column(String, index=True)
    url = Column(String)

class CodeSnippet(Base):
    __tablename__ = "snippets"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    filepath = Column(String, index=True)
    repo_id = Column(Integer, ForeignKey("repositories.id"))
    content = Column(String)
    chunk_index = Column(Integer)
    total_chunks = Column(Integer)
    start_line = Column(Integer)
    end_line = Column(Integer)
    embedding_id = Column(Integer, index=True)

Base.metadata.create_all(bind=engine)

# API Setup
app = FastAPI()

# Pydantic Schemas
class SearchQuery(BaseModel):
    query: str
    repo_name: str = None
    folder: str = None
    filters: dict = {}

def chunk_file(content, max_lines=MAX_CHUNK_LINES, overlap=OVERLAP_LINES):
    """Split file content into overlapping chunks"""
    lines = content.split('\n')
    chunks = []
    start = 0
    
    while start < len(lines):
        end = min(start + max_lines, len(lines))
        chunk = '\n'.join(lines[start:end])
        
        chunks.append({
            'content': chunk,
            'start_line': start + 1,
            'end_line': end
        })
        
        # Move with overlap
        start = end - overlap if end < len(lines) else end
    
    return chunks

def process_file(file_path, repo_path, repo_name):
    """Process a file into chunks with metadata"""
    try:
        filename = os.path.basename(file_path)
        file_ext = os.path.splitext(filename)[1]
        
        if filename in BLACKLIST_FILES:
            return []
        if file_ext in BLACKLIST_EXTENSIONS or file_ext not in VALID_EXTENSIONS:
            return []
        
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        chunks = chunk_file(content)
        relative_path = os.path.relpath(file_path, repo_path)
        
        return [{
            "filename": filename,
            "filepath": relative_path,
            "repo": repo_name,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "content": chunk['content'],
            "start_line": chunk['start_line'],
            "end_line": chunk['end_line']
        } for i, chunk in enumerate(chunks)]
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return []

@app.post("/repos/{repo_owner}/{repo_name}")
def clone_repo(repo_owner: str, repo_name: str, db: Session = Depends(get_db)):
    """Schedules repository cloning and indexing as a background task."""
    repo_url = f"https://github.com/{repo_owner}/{repo_name}.git"
    repo = Repository(name=repo_name, owner=repo_owner, url=repo_url)
    db.add(repo)
    db.commit()
    task_id = str(uuid.uuid4())
    clone_and_index_repo.apply_async(args=[repo_owner, repo_name, task_id])
    return {"message": "Repo added for processing", "repo": repo_url, "task_id": task_id}

@app.get("/task-status/{task_id}")
def get_task_status(task_id: str):
    """Returns the current status of a Celery task."""
    status = redis_client.get(task_id)
    if status:
        return {"task_id": task_id, "status": status.decode("utf-8")}
    return {"task_id": task_id, "status": "UNKNOWN"}

@app.post("/search")
def search_code(query: SearchQuery, db: Session = Depends(get_db)):
    """Searches code snippets using FAISS with repo and folder filtering."""
    start_time = time.time()
    print("FAISS index size:", index.ntotal)
    # print("Embedding shape:", embedding.shape)

    cache_key = f"search:{query.query}:{query.repo_name}:{query.folder}:{str(query.filters)}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return pickle.loads(cached_result)
    
    query_embedding = embedding_model.encode(query.query, convert_to_numpy=True).astype("float32").reshape(1, -1)
    D, I = index.search(query_embedding, 10)
    print("Query embedding shape:", query_embedding.shape)
    print("FAISS distances:", D)
    print("FAISS indices:", I)
    results = []
    
    for idx in I[0]:
        if idx in embedding_cache:
            snippet = embedding_cache[idx]
            if query.repo_name and snippet["repo"] != query.repo_name:
                continue
            if query.folder and not snippet["filepath"].startswith(query.folder):
                continue
            if all(query.filters.get(key, snippet[key]) == snippet[key] for key in query.filters):
                results.append({"filename": snippet["filename"], "filepath": snippet["filepath"], "content": snippet["content"], "repo": snippet["repo"]})
    
    response = {"results": results, "time_taken": time.time() - start_time}
    redis_client.set(cache_key, pickle.dumps(response), ex=3600)
    return response


@celery.task
@celery.task
def clone_and_index_repo(repo_owner: str, repo_name: str, task_id: str):
    redis_client.set(task_id, "IN_PROGRESS")
    repo_path = f"./repos/{repo_owner}_{repo_name}"
    
    try:
        # Clone repository (existing logic)
        if not os.path.exists(repo_path):
            os.makedirs(repo_path)
            git.Repo.clone_from(f"https://github.com/{repo_owner}/{repo_name}.git", repo_path)
        
        # Collect valid chunks in parallel
        with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
            futures = []
            valid_chunks = []
            
            for root, dirs, files in os.walk(repo_path):
                dirs[:] = [d for d in dirs if d not in BLACKLIST_FOLDERS]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    futures.append(executor.submit(
                        process_file, 
                        file_path, 
                        repo_path, 
                        repo_name
                    ))
            
            # Collect results
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                chunks = future.result()
                valid_chunks.extend(chunks)
        
        # Process in batches for embeddings
        total_chunks = len(valid_chunks)
        embeddings_buffer = []
        metadata_buffer = []
        
        for i in tqdm(range(0, total_chunks, BATCH_SIZE), desc="Generating embeddings"):
            batch = valid_chunks[i:i+BATCH_SIZE]
            batch_texts = [item["content"] for item in batch]
            
            batch_embeddings = embedding_model.encode(
                batch_texts,
                batch_size=MODEL_BATCH_SIZE,
                convert_to_numpy=True,
                show_progress_bar=False
            ).astype("float32")
            
            embeddings_buffer.append(batch_embeddings)
            metadata_buffer.extend(batch)
        
        # Bulk update FAISS index
        if embeddings_buffer:
            all_embeddings = np.concatenate(embeddings_buffer)
            index.add(all_embeddings)
            
            # Update embedding cache
            start_idx = len(embedding_cache)
            for i, metadata in enumerate(metadata_buffer):
                embedding_cache[start_idx + i] = metadata
        
        # Save index and cache
        faiss.write_index(index, faiss_index_path)
        with open(cached_embeddings_path, "wb") as f:
            pickle.dump(embedding_cache, f)
        
        redis_client.set(task_id, "COMPLETED")
    
    except Exception as e:
        redis_client.set(task_id, f"FAILED: {str(e)}")
        raise


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
