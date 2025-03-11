import httpx
import asyncio
import base64
import chardet
import os
import time
from services.embeddings import store_embeddings
from typing import List, Dict

GITHUB_API_URL = "https://api.github.com"
BATCH_SIZE = 10  # Process files in parallel
MAX_CONCURRENT_REQUESTS = 5  # Control how many requests happen at once
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Ignored Extensions, Folders, and Files
IGNORED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".ico", ".webp", ".tiff",
                      ".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv",
                      ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a",
                      ".ttf", ".otf", ".woff", ".woff2",
                      ".exe", ".dll", ".so", ".dylib", ".a", ".o",
                      ".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz",
                      ".iso", ".img", ".vmdk", ".bak", ".tmp", ".old",
                      ".pyc", ".pyo", ".pyd",
                      ".class", ".jar", ".war", ".ear", ".iml",
                      ".obj", ".lib", ".la",
                      ".log", ".out", ".err", "nohup.out",
                      ".cache", ".DS_Store", "Thumbs.db", "desktop.ini",
                      ".csv", ".tsv", ".jsonl", ".sqlite", ".db", ".mdb",
                      ".h5", ".pb", ".ckpt", ".onnx", ".tflite", ".pt",
                      ".patch", ".diff", ".rej",
                      ".min.js", ".min.css",
                      ".bundle.js", ".bundle.css",
                      "package-lock.json", "yarn.lock", ".pnp.js", ".pnp.cjs",
                      "Pipfile.lock", "Cargo.lock", "go.sum",
                      ".csproj", ".vbproj", ".nuspec",
                      ".gradle", ".mvn", ".map", ".br"}

IGNORED_FOLDERS = {".git", ".github", ".idea", ".vscode", "venv", ".venv", "env",
                   "node_modules", "bower_components", "dist", "build", "out",
                   "target", "bin", "obj", "lib", "__pycache__", "coverage",
                   ".pytest_cache", "logs", "cache", "tmp", ".gradle", ".mvn",
                   ".cargo", ".yarn", ".pnp", ".terraform", ".serverless",
                   ".next", ".nuxt", ".expo", ".angular",
                   ".yarn/cache", ".yarn/plugins", ".yarn/sdks", ".yarn/patches"}

IGNORED_FILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml",
                 "poetry.lock", "Pipfile.lock", "Cargo.lock", "go.sum",
                 ".editorconfig", ".eslintcache", ".prettierignore",
                 "sonar-project.properties", "codecov.yml",
                 "nohup.out", "error.log", "debug.log",
                 "Thumbs.db", "desktop.ini", ".DS_Store"}


async def get_repo_files(owner: str, repo: str, branch: str, token: str) -> List[str]:
    """Fetch all files in a GitHub repository with rate limit handling."""
    headers = {"Authorization": f"token {token}"}
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
    
    if response.status_code == 403:  # Rate-limited
        reset_time = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
        wait_time = reset_time - time.time()
        print(f"⏳ Rate limit reached. Waiting {wait_time:.2f} seconds...")
        time.sleep(wait_time + 1)  # Sleep till reset
        return await get_repo_files(owner, repo, branch, token)  # Retry after waiting
    
    if response.status_code != 200:
        raise Exception(f"GitHub API Error: {response.json()}")

    tree = response.json().get("tree", [])
    return [item["path"] for item in tree if item["type"] == "blob"]

async def get_file_content(owner: str, repo: str, file_path: str, token: str) -> str:
    """Fetch file content from GitHub with retries and encoding handling."""
    headers = {"Authorization": f"token {token}"}
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/contents/{file_path}"
    
    for attempt in range(3):  # Retry up to 3 times
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=headers, timeout=15)
                
                if response.status_code == 403:  # Rate limit handling
                    reset_time = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
                    wait_time = reset_time - time.time()
                    print(f"⏳ Rate limit reached. Waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time + 1)
                    continue  # Retry after waiting
                
                if response.status_code == 200:
                    content_b64 = response.json().get("content", "")
                    decoded_bytes = base64.b64decode(content_b64)
                    encoding = chardet.detect(decoded_bytes).get("encoding") or "utf-8"
                    return decoded_bytes.decode(encoding, errors="replace")
                
                print(f"⚠️ Error {response.status_code} fetching {file_path}, retrying ({attempt+1}/3)...")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            except Exception as e:
                print(f"❌ Exception fetching {file_path}: {e}")
                await asyncio.sleep(2)
    
    return None  # Failed after retries


async def process_repo_files(owner: str, repo: str, branch: str, token: str):
    """Fetches and processes repo files asynchronously."""
    files = await get_repo_files(owner, repo, branch, token)

    valid_files = [f for f in files if os.path.splitext(f)[1].lower() not in IGNORED_EXTENSIONS
                   and not any(folder in f for folder in IGNORED_FOLDERS)
                   and os.path.basename(f) not in IGNORED_FILES]
    print(f"📂 Found {len(valid_files)} files to process...")

    async def process_file(file_path):
        """Fetch content and store embeddings for a file while limiting concurrency."""
        async with semaphore:  # Limits concurrent requests
            print(f"📥 Fetching content for: {file_path}")  # Debug print
            file_content = await get_file_content(owner, repo, file_path, token)
            if file_content:
                print(f"✅ Fetched {len(file_content)} chars from {file_path}")
                await store_embeddings(owner, repo, file_path, file_content)
            else:
                print(f"⚠️ Skipped {file_path} (empty or error)")

            del file_content  # Free memory

    for i in range(0, len(valid_files), BATCH_SIZE):
        batch = valid_files[i:i + BATCH_SIZE]
        print(f"🚀 Processing batch {i // BATCH_SIZE + 1}/{(len(valid_files) // BATCH_SIZE) + 1}...")
        try:
            await asyncio.wait_for(
                asyncio.gather(*(process_file(file) for file in batch)),
                timeout=60  # Timeout after 60 seconds
            )
        except asyncio.TimeoutError:
            print("⚠️ Batch processing timed out! Skipping to the next batch...")

    print(f"✅ Successfully indexed {len(valid_files)} files from {repo}")
