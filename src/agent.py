import re
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tools import get_tool_prompt, AVAILABLE_TOOLS
except ImportError:
    from src.tools import get_tool_prompt, AVAILABLE_TOOLS

MODEL_NAME = "./models/Qwen2.5-Coder-1.5B-Agent"

_model = None
_tokenizer = None

def load_agent_model(model_name=MODEL_NAME):
    """Loads the model and tokenizer lazily."""
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    resolved_model = model_name
    if not os.path.exists(model_name):
        print(f"Warning: Local model path '{model_name}' not found. Falling back to base HuggingFace model.")
        resolved_model = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    print(f"Loading autonomous agent model ({resolved_model})...")
    _tokenizer = AutoTokenizer.from_pretrained(resolved_model)
    _model = AutoModelForCausalLM.from_pretrained(
        resolved_model,
        torch_dtype="auto",
        device_map="auto"
    )
    print("Agent ready! Provide a task or type 'exit' to quit.")
    return _model, _tokenizer

def parse_tool_call(text):
    """Extracts a JSON payload from <tool_call> tags or markdown json blocks, or parses markdown code blocks."""
    # 1. Intercept markdown code blocks for writing files
    code_match = re.search(r"`?([a-zA-Z0-9_\-/\\]+\.[a-zA-Z0-9]+)`?\s*:\s*\n?```[a-z]*\n(.*?)\n```", text, re.DOTALL)
    if code_match:
        return {
            "name": "write_file",
            "arguments": {
                "filepath": code_match.group(1).strip(),
                "content": code_match.group(2)
            }
        }
        
    # 2. Try exact tags first
    match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if not match:
        # 3. Fallback for small models: look for a json code block containing a tool call
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        
    if match:
        try:
            payload = json.loads(match.group(1), strict=False)
            if "name" in payload and "arguments" in payload:
                return payload
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON format in tool call: {e}"}
            
    # 4. Fallback for Qwen native tool call format (FunctionName\n{...})
    match_qwen = re.search(r"([A-Za-z_]+)\s*\n\s*(\{.*?\})", text, re.DOTALL)
    if match_qwen:
        tool_name = match_qwen.group(1).lower()
        try:
            args = json.loads(match_qwen.group(2), strict=False)
            return {"name": tool_name, "arguments": args}
        except:
            pass
            
    # 5. Ultimate Fallback: Catch raw code blocks (both ``` and <code>)
    match_code = re.search(r"(?:```[a-z]*\n|<code>\n)(.*?)(?:\n```|\n</code>)", text, re.DOTALL)
    if match_code:
        content = match_code.group(1)
        filename = "output.txt"
        name_match = re.search(r"(?:save|file).*?(?:as|to|name)?\s*[\*\`]?([a-zA-Z0-9_\-/\\]+\.[a-zA-Z0-9]+)[\*\`]?", text, re.IGNORECASE)
        if name_match:
            filename = name_match.group(1)
        elif "<html" in content.lower():
            filename = "index.html"
            
        return {
            "name": "write_file",
            "arguments": {
                "filepath": filename,
                "content": content
            }
        }
            
    return None

def execute_agent_loop(task, persona="Coder"):
    model, tokenizer = load_agent_model()
    system_prompt = get_tool_prompt(persona)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task}
    ]
    
    max_iterations = 10
    iteration = 0
    
    while iteration < max_iterations:
        print(f"\n--- Agent Thinking (Step {iteration + 1}) ---")
        
        # Format the chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Generate the response
        outputs = model.generate(
            **inputs, 
            max_new_tokens=1024,
            pad_token_id=tokenizer.eos_token_id
        )
        response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        
        print(f"\n[Agent]:\n{response_text}")
        messages.append({"role": "assistant", "content": response_text})
        
        # Parse for tool calls
        tool_call = parse_tool_call(response_text)
        
        if tool_call:
            tool_name = "System"
            if "error" in tool_call:
                result = tool_call["error"]
                print(f"\n[System]: {result}")
            else:
                tool_name = tool_call.get("name", "unknown")
                tool_args = tool_call.get("arguments", {})
                
                print(f"\n[System]: Executing tool '{tool_name}' with args: {tool_args}")
                
                if tool_name in AVAILABLE_TOOLS:
                    try:
                        # Unpack arguments and call the tool
                        result = AVAILABLE_TOOLS[tool_name](**tool_args)
                    except Exception as e:
                        result = f"Tool execution failed: {e}"
                else:
                    result = f"Error: Tool '{tool_name}' not found."
                    
            print(f"[Tool Output]:\n{result}")
            
            # Feed the observation back to the agent as a user message
            messages.append({"role": "user", "content": f"Observation from {tool_name}:\n{result}"})
            iteration += 1
        else:
            # If no tool call is made, assume the task is finished
            print("\n[System]: Task Finished.")
            break
            
    if iteration == max_iterations:
        print("\n[System]: Reached maximum iterations. Aborting task.")

if __name__ == "__main__":
    while True:
        try:
            user_task = input("\nTask: ")
            if user_task.lower() in ["exit", "quit"]:
                break
            if not user_task.strip():
                continue
            execute_agent_loop(user_task)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Critical Error: {e}")
