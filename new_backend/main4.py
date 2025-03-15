from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import faiss
import numpy as np
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from redis.exceptions import LockNotOwnedError, LockError
import threading
from celery import Celery
import git
import os
import redis
import uuid
import pickle
import time
import torch
import glob
import subprocess

# --------------------------
# Database Setup
# --------------------------
DATABASE_URL = "sqlite:///./codecompass.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --------------------------
# Directories & Constants
# --------------------------
INDEXES_DIR = "./indexes"
if not os.path.exists(INDEXES_DIR):
    os.makedirs(INDEXES_DIR)

# Embedding dimensions and batch sizes
d = 384  # Dimension of embeddings (for all-MiniLM-L6-v2)
BATCH_SIZE = 512       # For batch processing (if needed)
MODEL_BATCH_SIZE = 64  # For transformer model batching
MAX_CHUNK_LINES = 400  # Lines per chunk
OVERLAP_LINES = 20     # Overlap between chunks

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

# --------------------------
# Model Setup
# --------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

# --------------------------
# Database Dependency
# --------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --------------------------
# Redis & Celery Setup
# --------------------------
redis_client = redis.Redis(host="127.0.0.1", port=6379, db=0)
celery = Celery("tasks", broker="redis://127.0.0.1:6379/0", backend="redis://127.0.0.1:6379/0")

# --------------------------
# SQLAlchemy Models
# --------------------------
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

# --------------------------
# In-memory Cache for Loaded Indexes
# --------------------------
loaded_indexes = {}

def load_repo_index(repo_key):
    """
    Loads the FAISS index and embedding cache for a given repository key from disk.
    Checks both default branch (hybrid index) and non-default branch (delta index) based on available files.
    Uses a version file to decide if the in-memory version is up-to-date.
    """
    index_file_hybrid = os.path.join(INDEXES_DIR, f"{repo_key}_hybrid.index")
    metadata_file_hybrid = os.path.join(INDEXES_DIR, f"{repo_key}_metadata.pkl")
    index_file_delta = os.path.join(INDEXES_DIR, f"{repo_key}_delta.index")
    metadata_file_delta = os.path.join(INDEXES_DIR, f"{repo_key}_delta_metadata.pkl")

    if os.path.exists(index_file_hybrid) and os.path.exists(metadata_file_hybrid):
        index_file = index_file_hybrid
        metadata_file = metadata_file_hybrid
        version_suffix = "_hybrid"
    elif os.path.exists(index_file_delta) and os.path.exists(metadata_file_delta):
        index_file = index_file_delta
        metadata_file = metadata_file_delta
        version_suffix = "_delta"
    else:
        return None, None

    version_file = os.path.join(INDEXES_DIR, f"{repo_key}{version_suffix}_version.txt")
    version = None
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            version = f.read().strip()
    if repo_key in loaded_indexes and loaded_indexes[repo_key]["version"] == version:
        return loaded_indexes[repo_key]["index"], loaded_indexes[repo_key]["embedding_cache"]

    idx = faiss.read_index(index_file)
    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)
        vector_mapping = metadata.get("vector_id_to_chunk", {})
    loaded_indexes[repo_key] = {"index": idx, "embedding_cache": vector_mapping, "version": version}
    return idx, vector_mapping

# --------------------------
# Utility Functions
# --------------------------
def extend_lock(lock, lock_token, stop_event):
    """Continuously extend the lock's TTL until signaled to stop."""
    while not stop_event.is_set():
        time.sleep(30)  # Extend every 30 seconds
        try:
            if lock.local.token != lock_token:
                break  # Stop if lock token changes (lock lost)
            lock.extend(60)  # Extend the lock by 60 seconds
        except (AttributeError, LockError):
            break  # Stop if the lock is lost

def chunk_file(content, max_lines=MAX_CHUNK_LINES, overlap=OVERLAP_LINES):
    """Split file content into overlapping chunks."""
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
        start = end - overlap if end < len(lines) else end
    return chunks

def process_file(file_path, repo_path, repo_name):
    """Process a file into chunks with metadata."""
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

# --------------------------
# CodeCompassIndexer: Hybrid FAISS Index with Incremental Updates
# --------------------------
class CodeCompassIndexer:
    def __init__(self, index_path, metadata_path, dimension, nlist=256):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dimension = dimension
        self.nlist = nlist  # number of clusters for IVF
        self.index = self._create_index()
        # Mapping from composite key ("filepath-chunk_index") to vector ID
        self.chunk_to_vector_id = {}
        # Reverse mapping: vector ID to snippet metadata
        self.vector_id_to_chunk = {}
        self._load_metadata()

    def _create_index(self):
        """Creates a hybrid FAISS index using IVF with HNSW as the quantizer."""
        if os.path.exists(self.index_path):
            return faiss.read_index(self.index_path)
        quantizer = faiss.IndexHNSWFlat(self.dimension, 32)
        index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, faiss.METRIC_L2)
        dummy_data = np.random.randn(1000, self.dimension).astype(np.float32)
        index.train(dummy_data)
        return index

    def _load_metadata(self):
        """Loads the mapping from disk if available."""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "rb") as f:
                data = pickle.load(f)
                self.chunk_to_vector_id = data.get("chunk_to_vector_id", {})
                self.vector_id_to_chunk = data.get("vector_id_to_chunk", {})
        else:
            self.chunk_to_vector_id = {}
            self.vector_id_to_chunk = {}

    def save_index(self):
        """Saves the FAISS index and mapping to disk."""
        faiss.write_index(self.index, self.index_path)
        data = {
            "chunk_to_vector_id": self.chunk_to_vector_id,
            "vector_id_to_chunk": self.vector_id_to_chunk
        }
        with open(self.metadata_path, "wb") as f:
            pickle.dump(data, f)

    def update_index(self, current_chunks, get_embedding):
        """
        Incrementally updates the FAISS index based on current chunks.
        Each chunk is identified by a composite key: f"{filepath}-{chunk_index}"
        """
        current_keys = {f"{chunk['filepath']}-{chunk['chunk_index']}" for chunk in current_chunks}
        outdated_keys = set(self.chunk_to_vector_id.keys()) - current_keys
        if outdated_keys:
            print(f"[Indexer] Removing {len(outdated_keys)} outdated vectors...")
            remove_ids = [self.chunk_to_vector_id[key] for key in outdated_keys]
            self.index.remove_ids(np.array(remove_ids, dtype=np.int64))
            for key in outdated_keys:
                vector_id = self.chunk_to_vector_id.pop(key)
                if vector_id in self.vector_id_to_chunk:
                    del self.vector_id_to_chunk[vector_id]
        new_chunks = [chunk for chunk in current_chunks if f"{chunk['filepath']}-{chunk['chunk_index']}" not in self.chunk_to_vector_id]
        if new_chunks:
            print(f"[Indexer] Adding {len(new_chunks)} new/modified chunks...")
            new_vectors = []
            new_ids = []
            next_id = max(self.vector_id_to_chunk.keys(), default=-1) + 1
            for chunk in new_chunks:
                emb = get_embedding(chunk)
                if emb is not None:
                    new_vectors.append(emb)
                    new_ids.append(next_id)
                    composite_key = f"{chunk['filepath']}-{chunk['chunk_index']}"
                    self.chunk_to_vector_id[composite_key] = next_id
                    self.vector_id_to_chunk[next_id] = chunk
                    next_id += 1
            if new_vectors:
                new_vectors = np.array(new_vectors, dtype=np.float32)
                self.index.add_with_ids(new_vectors, np.array(new_ids, dtype=np.int64))
        self.save_index()
        print("[Indexer] FAISS index updated successfully.")

    def search(self, query_vector, top_k=10):
        """Searches the FAISS index using the hybrid settings."""
        self.index.nprobe = 10
        distances, indices = self.index.search(query_vector, top_k)
        return indices, distances

# --------------------------
# FastAPI Setup and Schemas
# --------------------------
app = FastAPI()

class SearchQuery(BaseModel):
    query: str
    repo_owner: str = None
    repo_name: str = None
    branch: str = None
    folder: str = None
    filters: dict = {}

# Updated indexing endpoint now accepts an optional branch parameter.
@app.post("/repos/{repo_owner}/{repo_name}")
def clone_repo(repo_owner: str, repo_name: str, branch: str = None, db: Session = Depends(get_db)):
    """
    Schedules repository cloning and indexing as a background task.
    The branch parameter (if provided) explicitly indicates which branch to index.
    """
    repo_url = f"https://github.com/{repo_owner}/{repo_name}.git"
    repo = Repository(name=repo_name, owner=repo_owner, url=repo_url)
    db.add(repo)
    db.commit()
    task_id = str(uuid.uuid4())
    # Pass the branch parameter to the Celery task.
    clone_and_index_repo.apply_async(args=[repo_owner, repo_name, task_id, branch])
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
    """
    Searches code snippets using FAISS.
    When repo_owner, repo_name and branch are provided, it only searches in that repository branch.
    """
    start_time = time.time()
    cache_key = f"search:{query.query}:{query.repo_owner}:{query.repo_name}:{query.branch}:{query.folder}:{str(query.filters)}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return pickle.loads(cached_result)
    
    query_embedding = embedding_model.encode(query.query, convert_to_numpy=True).astype("float32").reshape(1, -1)
    results = []
    
    if query.repo_owner and query.repo_name:
        base_prefix = f"{query.repo_owner}_{query.repo_name}"
        if query.branch:
            repo_key_target = f"{base_prefix}_{query.branch}"
            matching_keys = [key for key in get_all_repo_keys() if key == repo_key_target]
        else:
            matching_keys = [key for key in get_all_repo_keys() if key.startswith(base_prefix)]
    else:
        matching_keys = get_all_repo_keys()

    print(matching_keys)
    for repo_key in matching_keys:
        idx, emb_cache = load_repo_index(repo_key)
        if idx is None:
            continue
        distances, indices = idx.search(query_embedding, 10)
        for idx_val in indices[0]:
            if idx_val in emb_cache:
                snippet = emb_cache[idx_val]
                if query.folder and not snippet["filepath"].startswith(query.folder):
                    continue
                if all(query.filters.get(key, snippet.get(key)) == snippet.get(key) for key in query.filters):
                    results.append({
                        "filename": snippet["filename"],
                        "filepath": snippet["filepath"],
                        "content": snippet["content"],
                        "repo": snippet["repo"]
                    })
    
    response = {"results": results, "time_taken": time.time() - start_time}
    redis_client.set(cache_key, pickle.dumps(response), ex=3600)
    return response

def get_all_repo_keys():
    """
    Returns a list of repository keys derived from index file names.
    This includes both default branch (hybrid) indexes and non-default branch (delta) indexes.
    """
    files = glob.glob(os.path.join(INDEXES_DIR, "*_hybrid.index")) + glob.glob(os.path.join(INDEXES_DIR, "*_delta.index"))
    repo_keys = []
    for f in files:
        basename = os.path.basename(f)
        if basename.endswith("_hybrid.index"):
            key = basename.replace("_hybrid.index", "")
        elif basename.endswith("_delta.index"):
            key = basename.replace("_delta.index", "")
        repo_keys.append(key)
    return repo_keys

# --------------------------
# Celery Task: Clone and Index Repository with Incremental Hybrid Indexing
# --------------------------
@celery.task
def clone_and_index_repo(repo_owner: str, repo_name: str, task_id: str, branch: str = None):
    redis_client.set(task_id, "IN_PROGRESS")
    repo_path = f"./repos/{repo_owner}_{repo_name}"
    repo_key = f"{repo_owner}_{repo_name}_{branch or 'default'}"

    lock = redis_client.lock(f"lock_{repo_key}", timeout=60)
    if lock.acquire(blocking=False):
        try:
            print(f"✅ Lock acquired for {repo_key}, starting renewal thread.")
            lock_token = lock.local.token
            stop_event = threading.Event()
            renewer = threading.Thread(target=extend_lock, args=(lock, lock_token, stop_event))
            renewer.daemon = True
            renewer.start()

            if not os.path.exists(repo_path):
                os.makedirs(repo_path)
                repo = git.Repo.clone_from(f"https://github.com/{repo_owner}/{repo_name}.git", repo_path, depth=1)
            else:
                repo = git.Repo(repo_path)
                repo.git.fetch("--all")
                repo.git.reset('--hard')

            default_branch = repo.git.symbolic_ref("refs/remotes/origin/HEAD").split("/")[-1]
            if branch is None:
                branch = default_branch

            available_branches = [head.name for head in repo.heads] + [ref.name.split("/")[-1] for ref in repo.remote().refs]
            if branch in available_branches:
                repo.git.checkout(branch)
            else:
                print(f"⚠️ Branch '{branch}' not found. Falling back to '{default_branch}'.")
                repo.git.checkout(default_branch)
                branch = default_branch
                # Update repo_key so that it matches the branch actually indexed.
                repo_key = f"{repo_owner}_{repo_name}_{branch}"

            repo.git.reset('--hard')
            repo.git.pull()

            valid_chunks = []
            with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
                futures = []
                for root, dirs, files in os.walk(repo_path):
                    dirs[:] = [d for d in dirs if d not in BLACKLIST_FOLDERS]
                    for file in files:
                        file_path = os.path.join(root, file)
                        futures.append(executor.submit(process_file, file_path, repo_path, repo_name))
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                    chunks = future.result()
                    valid_chunks.extend(chunks)

            def get_chunk_embedding(chunk):
                try:
                    return embedding_model.encode(chunk["content"], convert_to_numpy=True).astype("float32")
                except Exception as e:
                    print(f"⚠️ Embedding failed: {e}. Retrying...")
                    return None

            # Use the actual branch name for naming the index/metadata files.
            if branch == default_branch:
                index_file = os.path.join(INDEXES_DIR, f"{repo_owner}_{repo_name}_{default_branch}_hybrid.index")
                metadata_file = os.path.join(INDEXES_DIR, f"{repo_owner}_{repo_name}_{default_branch}_metadata.pkl")
            else:
                index_file = os.path.join(INDEXES_DIR, f"{repo_key}_delta.index")
                metadata_file = os.path.join(INDEXES_DIR, f"{repo_key}_delta_metadata.pkl")

            version_file = os.path.join(INDEXES_DIR, f"{repo_owner}_{repo_name}_{branch}_version.txt")
            indexer = CodeCompassIndexer(index_file, metadata_file, d, nlist=256)
            indexer.update_index(valid_chunks, get_chunk_embedding)

            stop_event.set()
            with open(version_file + ".tmp", "w") as f:
                f.write(str(time.time()))
            os.replace(version_file + ".tmp", version_file)

            if repo_key in loaded_indexes:
                del loaded_indexes[repo_key]

            redis_client.set(task_id, "COMPLETED")
        except Exception as e:
            redis_client.set(task_id, f"FAILED: {str(e)}")
            raise
        finally:
            try:
                if lock.reacquire():
                    lock.release()
            except LockNotOwnedError:
                pass
    else:
        print(f"❌ Another process is already running this task: {repo_key}")

# --------------------------
# Main Application Runner
# --------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
