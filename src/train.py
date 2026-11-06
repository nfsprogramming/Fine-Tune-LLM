import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# 1. Load dataset
data_path = "data/agentic_dataset_hf.jsonl" if os.path.exists("data/agentic_dataset_hf.jsonl") else "agentic_dataset_hf.jsonl"
dataset = load_dataset("json", data_files=data_path, split="train")

# 2. Configure 4-bit Quantization (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 # bfloat16 for Ada Lovelace GPUs
)

# 3. Load Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# 4. Prepare model for k-bit training and configure LoRA
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 5. Format function for SFTTrainer
def format_prompts(example):
    text = []
    # apply_chat_template handles list of messages
    for msgs in example['messages']:
        text.append(tokenizer.apply_chat_template(msgs, tokenize=False))
    return {"text": text}

dataset = dataset.map(format_prompts, batched=True)

# 6. Configure Training Arguments
training_args = SFTConfig(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=1,
    optim="paged_adamw_32bit",
    bf16=True,
    fp16=False,
    save_steps=100,
    dataset_text_field="text",
    max_length=1024,
)

# 7. Train using SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)

print("Starting QLoRA Fine-Tuning...")
trainer.train()

# 8. Save the adapter
print("Saving adapter weights...")
adapter_dir = "./adapter" if os.path.isdir("./adapter") else "./qlora-agent-adapter"
trainer.model.save_pretrained(adapter_dir)
print("Done!")
