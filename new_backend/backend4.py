from fastapi import FastAPI, Depends, HTTPException
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
import threading
from collections import OrderedDict
import shutil

# Database Setup
DATABASE_URL = "sqlite:///./codecompass.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_BATCH_SIZE = 128
MAX_CHUNK_LINES = 400
OVERLAP_LINES = 20
REPO_BASE_PATH = "./repos"

# Load Sentence Transformer Model
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

# Redis Setup
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
    index_path = Column(String)
    embeddings_path = Column(String)

class CodeSnippet(Base):
    __tablename__ = "snippets"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    filepath = Column(String, index=True)
    repo_id = Column(Integer, ForeignKey("repositories.id"))
    content = Column(String)
    start_line = Column(Integer)
    end_line = Column(Integer)
    embedding_id = Column(Integer, index=True)

Base.metadata.create_all(bind=engine)

# Index Cache Manager
class IndexCache:
    def __init__(self, max_size=5):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def get_index(self, index_path):
        with self.lock:
            if index_path not in self.cache:
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)
                index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
                self.cache[index_path] = index
                self.cache.move_to_end(index_path)
            return self.cache[index_path]

index_cache = IndexCache(max_size=5)

# API Setup
app = FastAPI()

# Pydantic Schemas
class SearchQuery(BaseModel):
    query: str
    repo_name: str = None
    folder: str = None
    filters: dict = {}
    max_results: int = 10

# Helper Functions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def valid_file(file_path):
    filename = os.path.basename(file_path)
    file_ext = os.path.splitext(filename)[1]
    folder_parts = file_path.split(os.sep)
    
    return (filename not in BLACKLIST_FILES and
            file_ext in VALID_EXTENSIONS and
            not any(part in BLACKLIST_FOLDERS for part in folder_parts) and
            file_ext not in BLACKLIST_EXTENSIONS)

def chunk_content(content):
    lines = content.split('\n')
    return [
        {'content': '\n'.join(lines[start:end]),
         'start_line': start + 1,
         'end_line': end}
        for start in range(0, len(lines), MAX_CHUNK_LINES - OVERLAP_LINES)
        if (end := min(start + MAX_CHUNK_LINES, len(lines)))
    ]

# API Endpoints
@app.post("/repos/{repo_owner}/{repo_name}")
def add_repo(repo_owner: str, repo_name: str, db: Session = Depends(get_db)):
    repo_url = f"https://github.com/{repo_owner}/{repo_name}.git"
    existing = db.query(Repository).filter_by(name=repo_name, owner=repo_owner).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Repository already exists")
    
    repo = Repository(
        name=repo_name,
        owner=repo_owner,
        url=repo_url,
        index_path="",
        embeddings_path=""
    )
    db.add(repo)
    db.commit()
    
    task_id = str(uuid.uuid4())
    clone_and_index_repo.apply_async(args=[repo.id, task_id])
    
    return {"message": "Repo queued for indexing", "task_id": task_id}

@app.get("/search")
def search_code(query: SearchQuery, db: Session = Depends(get_db)):
    start_time = time.time()
    query_embedding = embedding_model.encode(
        query.query,
        batch_size=MODEL_BATCH_SIZE,
        convert_to_numpy=True
    ).astype("float32").reshape(1, -1)

    results = []
    query_filter = [Repository.name == query.repo_name] if query.repo_name else []
    
    repos = db.query(Repository).filter(*query_filter).all()
    
    for repo in repos:
        try:
            index = index_cache.get_index(repo.index_path)
            
            with open(repo.embeddings_path, "rb") as f:
                embeddings_meta = pickle.load(f)
            
            scores, indices = index.search(query_embedding, query.max_results)
            
            for idx, score in zip(indices[0], scores[0]):
                if idx < 0 or idx >= len(embeddings_meta):
                    continue
                
                meta = embeddings_meta[idx]
                if query.folder and not meta['filepath'].startswith(query.folder):
                    continue
                
                snippet = db.query(CodeSnippet).get(meta['snippet_id'])
                results.append({
                    "score": float(score),
                    "content": snippet.content,
                    "filepath": meta['filepath'],
                    "repo": repo.name,
                    "start_line": meta['start_line'],
                    "end_line": meta['end_line']
                })
        
        except Exception as e:
            print(f"Error searching {repo.name}: {str(e)}")
    
    results.sort(key=lambda x: x['score'])
    return {
        "results": results[:query.max_results],
        "time": time.time() - start_time
    }

# Celery Task
@celery.task
def clone_and_index_repo(repo_id: int, task_id: str):
    db = SessionLocal()
    redis_client.set(task_id, "IN_PROGRESS")
    
    try:
        repo = db.query(Repository).get(repo_id)
        if not repo:
            raise ValueError("Repository not found")
        
        repo_path = os.path.join(REPO_BASE_PATH, f"{repo.owner}_{repo.name}")
        index_path = f"index_{repo.id}.faiss"
        embeddings_path = f"embeddings_{repo.id}.pkl"

        # Clean existing data
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)
        
        # Clone fresh copy
        git.Repo.clone_from(repo.url, repo_path)
        
        # Delete old snippets
        db.query(CodeSnippet).filter(CodeSnippet.repo_id == repo.id).delete()
        
        # File processing
        valid_chunks = []
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            
            for root, _, files in os.walk(repo_path):
                if any(part in BLACKLIST_FOLDERS for part in root.split(os.sep)):
                    continue
                
                for file in files:
                    file_path = os.path.join(root, file)
                    if valid_file(file_path):
                        futures.append(executor.submit(
                            process_file,
                            file_path,
                            repo_path,
                            repo.id,
                            db
                        ))
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                valid_chunks.extend(future.result())
        
        # Create FAISS index
        index = faiss.IndexHNSWFlat(384, 32)
        index.hnsw.efConstruction = 40
        
        embeddings = []
        embeddings_meta = []
        
        for chunk in valid_chunks:
            embedding = embedding_model.encode(
                chunk['content'],
                convert_to_numpy=True
            ).astype("float32")
            
            embeddings.append(embedding)
            embeddings_meta.append({
                'snippet_id': chunk['id'],
                'filepath': chunk['filepath'],
                'start_line': chunk['start_line'],
                'end_line': chunk['end_line']
            })
        
        if embeddings:
            index.add(np.array(embeddings))
            faiss.write_index(index, index_path)
            
            with open(embeddings_path, "wb") as f:
                pickle.dump(embeddings_meta, f)
            
            repo.index_path = index_path
            repo.embeddings_path = embeddings_path
            db.commit()
        
        redis_client.set(task_id, "COMPLETED")
    
    except Exception as e:
        db.rollback()
        redis_client.set(task_id, f"FAILED: {str(e)}")
        raise
    
    finally:
        db.close()

def process_file(file_path: str, repo_path: str, repo_id: int, db: Session):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        chunks = chunk_content(content)
        relative_path = os.path.relpath(file_path, repo_path)
        snippets = []
        
        for chunk in chunks:
            snippet = CodeSnippet(
                filename=os.path.basename(file_path),
                filepath=relative_path,
                repo_id=repo_id,
                content=chunk['content'],
                start_line=chunk['start_line'],
                end_line=chunk['end_line']
            )
            db.add(snippet)
            db.commit()
            db.refresh(snippet)
            
            snippets.append({
                'id': snippet.id,
                'filepath': relative_path,
                'start_line': chunk['start_line'],
                'end_line': chunk['end_line']
            })
        
        return snippets
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return []

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)