import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, AutoModel
from qdrant_client import QdrantClient

# Configuration
QWEN_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
EMBED_MODEL_PATH = "./models/bge-small-en-v1.5"
QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "codebase"
TOP_K = 3

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading embedding model ({EMBED_MODEL_ID})...")
embed_load_path = EMBED_MODEL_PATH if os.path.exists(EMBED_MODEL_PATH) else EMBED_MODEL_ID
embed_tokenizer = AutoTokenizer.from_pretrained(embed_load_path)
embed_model = AutoModel.from_pretrained(embed_load_path)
embed_model.eval()
embed_model.to(device)

print(f"Loading generative model ({QWEN_MODEL})...")
gen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
gen_model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL,
    torch_dtype="auto",
    device_map="auto"
)

print("Connecting to Qdrant...")
try:
    qdrant = QdrantClient(path=QDRANT_PATH)
    # Check if collection exists
    collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in collections:
        print(f"Warning: Collection '{COLLECTION_NAME}' not found. Did you run indexer.py?")
except Exception as e:
    print(f"Qdrant connection error: {e}")
    qdrant = None

def get_query_embedding(text):
    inputs = embed_tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = embed_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings[0].cpu().tolist()

print("\nModel loaded. Type 'exit' or 'quit' to stop.")

while True:
    try:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input.strip():
            continue

        context_str = ""
        if qdrant:
            try:
                # 1. Embed the user's query
                query_vector = get_query_embedding(user_input)
                
                # 2. Search Qdrant
                search_result = qdrant.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_vector,
                    limit=TOP_K
                )
                
                # 3. Format the context
                if search_result:
                    context_str = "Here is some relevant context from the user's codebase:\n\n"
                    for i, hit in enumerate(search_result):
                        filepath = hit.payload.get("filepath", "Unknown")
                        content = hit.payload.get("content", "")
                        context_str += f"--- File: {filepath} ---\n{content}\n\n"
            except Exception as e:
                print(f"[RAG Error: {e}]")

        system_prompt = "You are a helpful AI coding assistant."
        if context_str:
            system_prompt += "\n" + context_str

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        text = gen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = gen_tokenizer([text], return_tensors="pt").to(gen_model.device)

        streamer = TextStreamer(gen_tokenizer, skip_prompt=True, skip_special_tokens=True)
        print(f"\nAssistant:")
        
        gen_model.generate(
            **model_inputs,
            max_new_tokens=1024,
            streamer=streamer
        )
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Error: {e}")
