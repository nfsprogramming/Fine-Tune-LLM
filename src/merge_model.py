import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def main():
    base_model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    adapter_path = "./adapter" if os.path.exists("./adapter/adapter_config.json") else "./qlora-agent-adapter"
    save_path = "./models/Qwen2.5-Coder-1.5B-Agent" # Save to a new folder to be safe

    print(f"1. Loading base model: {base_model_id}")
    # Note: We load in bfloat16 to merge the weights. We cannot merge weights while in 4-bit mode.
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print(f"2. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    print(f"3. Attaching LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("4. Merging adapter weights permanently into the base model...")
    # This mathematically adds the LoRA matrix back into the base model's attention weights
    merged_model = model.merge_and_unload()

    print(f"5. Saving the new standalone model to {save_path}...")
    os.makedirs(save_path, exist_ok=True)
    merged_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print("Done! You now have a permanently fine-tuned, standalone model.")

if __name__ == "__main__":
    main()
