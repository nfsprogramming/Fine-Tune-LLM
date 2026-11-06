import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
adapter_path = "./adapter" if os.path.exists("./adapter/adapter_config.json") else "./qlora-adapter"

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("Evaluation setup complete. Type 'exit' to quit.")

while True:
    try:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input.strip():
            continue

        messages = [
            {"role": "system", "content": "You are a helpful AI coding assistant."},
            {"role": "user", "content": user_input}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=200)
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        print(f"\nAssistant:\n{response}")
        
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Error: {e}")
