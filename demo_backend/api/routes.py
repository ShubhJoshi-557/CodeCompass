from fastapi import APIRouter, HTTPException, BackgroundTasks
from models.repo_requests import RepoRequest
from services.github import process_repo_files, get_repo_files, get_file_content
from services.chromadb import collection
from services.embeddings import embedding_model
import asyncio

router = APIRouter()

async def background_indexing(owner: str, repo: str, branch: str, token: str):
    """Background task to index repo asynchronously."""
    try:
        task = asyncio.create_task(process_repo_files(owner, repo, branch, token))
        await task   
        print(f"✅ Indexing complete for {owner}/{repo}")
    except Exception as e:
        print(f"❌ Error during indexing: {str(e)}")

@router.post("/fetch-files")
async def fetch_files(request: RepoRequest):
    """Fetch repo files and return contents."""
    try:
        files = await get_repo_files(request.owner, request.repo, request.branch, request.token)
        file_contents = {file: await get_file_content(request.owner, request.repo, file, request.token) for file in files if file is not None}
        return {"files": file_contents}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/index_repo/")
async def index_repo(owner: str, repo: str, branch: str, token: str, background_tasks: BackgroundTasks):
    """Trigger async repo indexing."""
    background_tasks.add_task(background_indexing, owner, repo, branch, token)
    return {"message": f"Indexing started for {owner}/{repo}. Check logs for progress."}

@router.post("/search/")
def search_code(query: str):
    """Search indexed code using embeddings."""
    if collection.count() == 0:
        return {"matches": [], "message": "No code indexed yet. Try indexing first."}

    query_embedding = embedding_model.encode([query], convert_to_numpy=True).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    
    return {"matches": results}
