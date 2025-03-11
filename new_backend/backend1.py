from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, Session
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

# Load Sentence Transformer Model for Code Search
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

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
def clone_and_index_repo(repo_owner: str, repo_name: str, task_id: str):
    redis_client.set(task_id, "IN_PROGRESS")
    repo_path = f"./repos/{repo_owner}_{repo_name}"
    if not os.path.exists(repo_path):
        os.makedirs(repo_path)
    
    try:
        # Check if the folder exists and is a valid Git repo
        if os.path.exists(repo_path) and os.path.isdir(repo_path):
            try:
                _ = git.Repo(repo_path).git_dir  # Check if it's a valid Git repository
                print(f"Repository {repo_name} already exists, skipping cloning.")
            except git.exc.InvalidGitRepositoryError:
                print(f"Directory exists but is not a valid git repo. Deleting and re-cloning.")
                os.system(f"rm -rf {repo_path}")  # Delete invalid directory
                git.Repo.clone_from(f"https://github.com/{repo_owner}/{repo_name}.git", repo_path)
        else:
            print(f"Cloning repository {repo_name}...")
            git.Repo.clone_from(f"https://github.com/{repo_owner}/{repo_name}.git", repo_path)

        # Valid source code file extensions
        valid_extensions = {
            ".py", ".js", ".jsx", ".ts", ".tsx", ".cpp", ".c", ".cc", ".h", ".hpp",
            ".java", ".cs", ".go", ".rs", ".swift", ".kt", ".kts", ".dart", ".rb",
            ".php", ".pl", ".pm", ".lua", ".r", ".sh", ".bash", ".zsh", ".fish",
            ".bat", ".cmd", ".ps1", ".vue", ".svelte", ".pug", ".jade", ".tf",
            ".sql", ".graphql", ".gql", ".gd", ".m", ".asm", ".rkt", ".ex", ".exs",
            ".erl", ".hrl", ".lisp", ".clj", ".cljs", ".cljc", ".ml", ".mli", ".f90",
            ".f95", ".fs", ".fsx", ".fsi", ".jl", ".nim", ".zig", ".ada", ".d", ".p",
            ".pro", ".idris", ".cr", ".v", ".sv", ".pde", ".q", ".sas", ".st", ".sml",
            "Dockerfile"
        }

        # Blacklist folders that contain non-source code files
        blacklist_folders = {
            "node_modules", "vendor", "bin", "obj", "dist", "build", "__pycache__",
            ".git", ".svn", ".hg", ".vscode", ".idea", ".vs", "target", "out", 
            "tmp", "cache", "logs", "__snapshots__", "__tests__", "test", "examples"
        }

        # Blacklist specific non-code files
        blacklist_files = {
            "package-lock.json", "yarn.lock", "Makefile", "README.md", "LICENSE",
            "CHANGELOG.md", ".DS_Store", "Thumbs.db", ".npmrc", ".yarnrc", "go.mod",
            "go.sum", "Cargo.lock", "Pipfile.lock", "poetry.lock"
        }

        # Blacklist non-source code file extensions
        blacklist_extensions = {
            ".json", ".xml", ".yaml", ".yml", ".lock", ".md", ".markdown", ".txt",
            ".ini", ".log", ".tmp", ".iml", ".sln", ".csproj", ".classpath",
            ".project", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".mp3",
            ".mp4", ".avi", ".pdf", ".docx", ".xlsx", ".zip", ".tar", ".gz", 
            ".7z", ".rar", ".bin", ".exe", ".dll", ".dylib", ".so"
        }

        # Traverse and index only valid files
        for root, dirs, files in os.walk(repo_path):
            # Exclude blacklisted folders
            dirs[:] = [d for d in dirs if d not in blacklist_folders]

            for file in files:
                if file in blacklist_files:
                    continue
                
                file_ext = os.path.splitext(file)[1]
                if file_ext in blacklist_extensions or file_ext not in valid_extensions:
                    continue
                
                file_path = os.path.join(root, file)
                
                # Process and index the file
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                embedding = embedding_model.encode(content, convert_to_numpy=True).astype("float32").reshape(1, -1)
                index.add(embedding)
                embedding_cache[len(embedding_cache)] = {"filename": file, "filepath": file_path, "content": content, "repo": repo_name}
        
        faiss.write_index(index, faiss_index_path)
        
        with open(cached_embeddings_path, "wb") as f:
            pickle.dump(embedding_cache, f)
        
        redis_client.set(task_id, "COMPLETED")
    
    except Exception as e:
        redis_client.set(task_id, f"FAILED: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
