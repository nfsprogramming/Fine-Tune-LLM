import os
import uuid
import torch
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Configuration
MODEL_ID = "BAAI/bge-small-en-v1.5"
MODEL_PATH = "./models/bge-small-en-v1.5"
QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "codebase"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

IGNORE_DIRS = {".venv", "node_modules", ".git", "__pycache__", "models", "qdrant_db", "docs"}
VALID_EXTENSIONS = {".py", ".js", ".ts", ".md", ".jsonl", ".txt"}

print("Loading embedding model...")
# Use local path if downloaded, else use model ID
load_path = MODEL_PATH if os.path.exists(MODEL_PATH) else MODEL_ID
tokenizer = AutoTokenizer.from_pretrained(load_path)
model = AutoModel.from_pretrained(load_path)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded on {device}")

print("Initializing Qdrant...")
client = QdrantClient(path=QDRANT_PATH)

# Create collection if it doesn't exist
collections = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME not in collections:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    print(f"Created new collection: {COLLECTION_NAME}")

def get_embeddings(text):
    # Prepare input for BAAI/bge-small-en-v1.5
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # Use CLS token for embedding representation
        embeddings = outputs.last_hidden_state[:, 0, :]
        # Normalize embeddings for cosine similarity
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings[0].cpu().tolist()

def chunk_text(text, chunk_size, overlap):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    i = 0
    if not tokens:
        return chunks
        
    while i < len(tokens):
        chunk_tokens = tokens[i:i+chunk_size]
        chunk_str = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_str)
        i += chunk_size - overlap
    return chunks

def index_codebase(root_dir):
    points = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Filter ignored directories in place
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith('.')]
        
        for file in filenames:
            ext = os.path.splitext(file)[1]
            if ext not in VALID_EXTENSIONS:
                continue
                
            filepath = os.path.join(dirpath, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"Skipping {filepath}: {e}")
                continue
                
            print(f"Indexing {filepath}...")
            chunks = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
            
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                vector = get_embeddings(chunk)
                
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "filepath": filepath,
                        "chunk_index": i,
                        "content": chunk
                    }
                )
                points.append(point)
                
                # Batch upsert every 50 chunks
                if len(points) >= 50:
                    client.upsert(collection_name=COLLECTION_NAME, points=points)
                    points = []
                    
    # Insert remaining points
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        
    print(f"Indexing complete! Database saved at {QDRANT_PATH}")

if __name__ == "__main__":
    print("Starting indexing process for current directory...")
    index_codebase(".")
