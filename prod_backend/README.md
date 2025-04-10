# 🧠 CodeCompass Backend

This is the backend for **CodeCompass**, an AI-powered codebase search and explanation engine. It leverages FastAPI, FAISS, Redis, Celery, Sentence Transformers, and Groq API for intelligent semantic code search and interaction.

---

## 🚀 Features

- FastAPI-based REST API
- SentenceTransformer embeddings + FAISS for fast semantic search
- Celery with Redis for asynchronous background task processing
- Git integration for codebase ingestion
- Groq API for powerful LLM-based responses

---

## 📆 Requirements

- Python 3.12.3
- Redis server (local or cloud)

Install all Python dependencies with:

```bash
pip install -r requirements.txt
```

### requirements.txt (with pinned versions)
```
amqp==5.3.1
annotated-types==0.7.0
anyio==4.9.0
billiard==4.2.1
celery==5.5.1
certifi==2025.1.31
charset-normalizer==3.4.1
click==8.1.8
click-didyoumean==0.3.1
click-plugins==1.1.1
click-repl==0.3.0
distro==1.9.0
faiss-cpu==1.10.0
fastapi==0.115.12
filelock==3.18.0
fsspec==2025.3.2
gitdb==4.0.12
GitPython==3.1.44
greenlet==3.1.1
groq==0.22.0
h11==0.14.0
httpcore==1.0.7
httpx==0.28.1
huggingface-hub==0.30.2
idna==3.10
Jinja2==3.1.6
joblib==1.4.2
kombu==5.5.2
MarkupSafe==3.0.2
mpmath==1.3.0
networkx==3.4.2
numpy==2.2.4
nvidia-cublas-cu12==12.4.5.8
nvidia-cuda-cupti-cu12==12.4.127
nvidia-cuda-nvrtc-cu12==12.4.127
nvidia-cuda-runtime-cu12==12.4.127
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.2.1.3
nvidia-curand-cu12==10.3.5.147
nvidia-cusolver-cu12==11.6.1.9
nvidia-cusparse-cu12==12.3.1.170
nvidia-cusparselt-cu12==0.6.2
nvidia-nccl-cu12==2.21.5
nvidia-nvjitlink-cu12==12.4.127
nvidia-nvtx-cu12==12.4.127
packaging==24.2
pillow==11.1.0
prompt_toolkit==3.0.50
pydantic==2.11.3
pydantic_core==2.33.1
python-dateutil==2.9.0.post0
python-dotenv==1.1.0
PyYAML==6.0.2
redis==5.2.1
regex==2024.11.6
requests==2.32.3
safetensors==0.5.3
scikit-learn==1.6.1
scipy==1.15.2
sentence-transformers==4.0.2
setuptools==78.1.0
six==1.17.0
smmap==5.0.2
sniffio==1.3.1
SQLAlchemy==2.0.40
starlette==0.46.1
sympy==1.13.1
threadpoolctl==3.6.0
tokenizers==0.21.1
torch==2.6.0
tqdm==4.67.1
transformers==4.51.1
triton==3.2.0
typing-inspection==0.4.0
typing_extensions==4.13.1
tzdata==2025.2
urllib3==2.3.0
uvicorn==0.34.0
vine==5.1.0
wcwidth==0.2.13
```

---

## 🔧 Setup Instructions

### 1. 📂 Clone the repository
```bash
git clone https://github.com/yourusername/codecompass-backend.git
cd codecompass-backend
```

### 2. 🧵 Create and activate a virtual environment

#### On Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. 📄 Setup the environment variables
Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key
REDIS_URL=redis://localhost:6379
```

---

## 🌐 Running the App

### 1. Start the API server
```bash
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Start the Celery worker
```bash
python3 -m celery -A main.celery worker --loglevel=info
```

---

## 🔍 Interactive API Docs

Once running, visit: [http://localhost:8000/docs](http://localhost:8000/docs)

This opens the Swagger UI with full API testing support.

---

## 🔧 Tips

- Use `pip freeze > requirements.txt` to regenerate dependencies.
- Use `redis-cli` to test or inspect your Redis cache.
- Add `venv/`, `__pycache__/`, `*.pyc`, `.env` to `.gitignore`.

---

## 💼 License

MIT License © 2025 Shubh

---

## 👤 Author

**Shubh**  
GitHub: [@yourusername](https://github.com/yourusername)  
LinkedIn: [your-profile](https://linkedin.com/in/your-profile)

---

