# import os
# import time
# import redis
# import faiss
# import psycopg2
# import numpy as np
# from flask import Flask, request, jsonify
# from sentence_transformers import SentenceTransformer

# # Initialize Flask app
# app = Flask(__name__)

# # Load embedding model
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Redis for caching
# redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# # FAISS index for vector search
# vector_dim = 384
# faiss_index = faiss.IndexFlatL2(vector_dim)

# # PostgreSQL for metadata storage
# conn = psycopg2.connect(
#     dbname="codecompass", user="user", password="password", host="localhost", port="5432"
# )
# cursor = conn.cursor()

# # Create table if not exists
# cursor.execute("""
# CREATE TABLE IF NOT EXISTS snippets (
#     id SERIAL PRIMARY KEY,
#     owner TEXT,
#     repo TEXT,
#     file_path TEXT,
#     filename TEXT,
#     content TEXT,
#     embedding VECTOR(384)
# );
# """)
# conn.commit()

# # Function to generate embeddings
# def get_embedding(text):
#     return embedding_model.encode(text).astype(np.float32)

# # Function to check cache
# def check_cache(query):
#     cached_result = redis_client.get(query)
#     if cached_result:
#         return jsonify(eval(cached_result))
#     return None

# # API to index a repository
# @app.route("/index", methods=["POST"])
# def index_repo():
#     data = request.json
#     owner = data["owner"]
#     repo = data["repo"]
#     base_path = data["base_path"]

#     all_files = []
#     for root, _, files in os.walk(base_path):
#         for file in files:
#             file_path = os.path.join(root, file)
#             with open(file_path, "r", errors="ignore") as f:
#                 content = f.read()
#             embedding = get_embedding(content)
#             all_files.append((owner, repo, file_path, file, content, embedding))

#     # Store in database
#     cursor.executemany("""
#     INSERT INTO snippets (owner, repo, file_path, filename, content, embedding)
#     VALUES (%s, %s, %s, %s, %s, %s);
#     """, [(o, r, p, f, c, e.tolist()) for o, r, p, f, c, e in all_files])
#     conn.commit()

#     # Store embeddings in FAISS
#     embeddings = np.array([entry[5] for entry in all_files])
#     faiss_index.add(embeddings)

#     return jsonify({"message": "Indexing complete"})

# # API to search code
# @app.route("/search", methods=["POST"])
# def search_code():
#     start_time = time.time()
#     data = request.json
#     query = data["query"]
#     filters = data.get("filters", {})

#     # Check cache
#     cached_response = check_cache(query)
#     if cached_response:
#         return cached_response

#     # Get query embedding
#     query_embedding = get_embedding(query).reshape(1, -1)

#     # FAISS search
#     _, indices = faiss_index.search(query_embedding, 10)

#     # Retrieve results from database
#     index_list = tuple(indices[0].tolist())
#     cursor.execute("""
#     SELECT owner, repo, file_path, filename, content
#     FROM snippets
#     WHERE id IN %s;
#     """, (index_list,))
#     results = cursor.fetchall()

#     # Apply filters
#     filtered_results = [
#         {
#             "owner": r[0], "repo": r[1], "file_path": r[2], "filename": r[3], "content": r[4]
#         }
#         for r in results
#         if r[0] == filters.get("owner", r[0])
#         and r[1] == filters.get("repo", r[1])
#         and r[2].startswith(filters.get("folder", ""))
#     ]

#     # Cache response
#     redis_client.setex(query, 3600, str(filtered_results))

#     response_time = time.time() - start_time
#     return jsonify({"results": filtered_results, "time_taken": response_time})

# if __name__ == "__main__":
#     app.run(debug=True)
