from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Download Qwen2.5-Coder models.")
    parser.add_argument("--size", type=str, default="1.5b", choices=["1.5b", "7b", "14b", "32b"], help="Model size to download")
    args = parser.parse_args()

    size_map = {
        "1.5b": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "14b": "Qwen/Qwen2.5-Coder-14B-Instruct",
        "32b": "Qwen/Qwen2.5-Coder-32B-Instruct"
    }
    
    model_id = size_map[args.size]
    save_path = f"./models/{model_id.split('/')[-1]}"

    print(f"Downloading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

    print(f"Saving to {save_path}...")
    os.makedirs(save_path, exist_ok=True)
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)

    print(f"Model successfully saved to {save_path}")

if __name__ == "__main__":
    main()
