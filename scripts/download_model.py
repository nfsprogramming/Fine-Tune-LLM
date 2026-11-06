from transformers import AutoTokenizer, AutoModel
import os

model_id = "BAAI/bge-small-en-v1.5"
save_path = "./models/bge-small-en-v1.5"

print(f"Downloading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

print(f"Saving to {save_path}...")
os.makedirs(save_path, exist_ok=True)
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print(f"Model successfully saved to {save_path}")
