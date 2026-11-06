import os
import json
try:
    from datasets import load_dataset
except ImportError:
    print("Please install datasets library: pip install datasets")
    exit(1)

def main():
    print("Downloading dataset from Hugging Face (pyromind/agentic-tool-call-dataset-12k)...")
    dataset = load_dataset("pyromind/agentic-tool-call-dataset-12k", "short", split="train")
    
    print(f"Total examples downloaded: {len(dataset)}")
    
    # Filter for short examples to avoid CUDA OOM on 6GB VRAM
    filtered_data = []
    for item in dataset:
        messages = item.get("messages", [])
        
        # Check token length by total characters
        total_len = sum(len(str(m.get("content", ""))) for m in messages)
        if total_len < 2000:
            # Enforce our system prompt
            if len(messages) > 0 and messages[0].get("role") == "system":
                messages[0]["content"] = "You are an autonomous AI software engineer acting as a Coder. To use a tool, you MUST output a JSON block wrapped in <tool_call> tags. Example: <tool_call>\n{\n  \"name\": \"write_file\",\n  \"arguments\": {\"filepath\": \"main.py\"}\n}\n</tool_call>"
            else:
                messages.insert(0, {"role": "system", "content": "You are an autonomous AI software engineer acting as a Coder. To use a tool, you MUST output a JSON block wrapped in <tool_call> tags. Example: <tool_call>\n{\n  \"name\": \"write_file\",\n  \"arguments\": {\"filepath\": \"main.py\"}\n}\n</tool_call>"})
            
            filtered_data.append({"messages": messages})
            
        if len(filtered_data) >= 1000: # We only need 1000 for local fine-tuning
            break
            
    print(f"Filtered down to {len(filtered_data)} short examples suitable for RTX 4050 (6GB VRAM).")
    
    output_path = os.path.join("data", "agentic_dataset_hf.jsonl") if os.path.isdir("data") else "agentic_dataset_hf.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in filtered_data:
            f.write(json.dumps(ex) + "\n")
            
    print(f"Saved formatted dataset to {output_path}!")

if __name__ == "__main__":
    main()
