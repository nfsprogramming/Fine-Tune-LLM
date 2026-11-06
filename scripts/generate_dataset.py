import os
import json
import random

SYSTEM_PROMPT = "You are an autonomous AI software engineer acting as a Coder...\n\nTo use a tool, you MUST output a JSON block wrapped in <tool_call> tags..."

def generate_bug_fix_scenario():
    """Generates a dataset entry for reading a file."""
    files = ["main.py", "app.js", "utils.go", "index.html"]
    file = random.choice(files)
    
    task = f"Fix the bug in {file}"
    thought = f"<thought>I need to read the contents of {file} to understand the bug before I can fix it.</thought>"
    tool_call = f"""```json\n{{\n  "name": "read_file",\n  "arguments": {{\n    "filepath": "{file}"\n  }}\n}}\n```"""
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
            {"role": "assistant", "content": f"{thought}\n{tool_call}"}
        ]
    }

def generate_file_write_scenario():
    """Generates a dataset entry using the markdown file writing syntax."""
    scenarios = [
        ("index.html", "<html><body>Hello</body></html>"),
        ("script.py", "print('Hello World')"),
        ("styles.css", "body { background: red; }")
    ]
    file, content = random.choice(scenarios)
    
    task = f"Create {file} with some basic content."
    thought = f"<thought>I will use the special markdown syntax to write to {file} directly.</thought>"
    response = f"{thought}\n\n{file}:\n```\n{content}\n```"
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
            {"role": "assistant", "content": response}
        ]
    }

def generate_git_scenario():
    """Generates a dataset entry for running git commands."""
    commands = ["git status", "git log -n 5", "git diff"]
    cmd = random.choice(commands)
    
    task = f"Check the current repository status using '{cmd}'."
    thought = f"<thought>I need to run {cmd} in the terminal to check the status.</thought>"
    tool_call = f"""```json\n{{\n  "name": "git_tool",\n  "arguments": {{\n    "command": "{cmd.replace('git ', '')}"\n  }}\n}}\n```"""
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
            {"role": "assistant", "content": f"{thought}\n{tool_call}"}
        ]
    }

def main():
    print("Generating agentic dataset...")
    dataset = []
    
    # Generate scenarios
    for _ in range(200):
        dataset.append(generate_bug_fix_scenario())
        dataset.append(generate_file_write_scenario())
        dataset.append(generate_git_scenario())
        
    random.shuffle(dataset)
    
    output_path = os.path.join("data", "agentic_dataset.jsonl") if os.path.isdir("data") else "agentic_dataset.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
            
    print(f"Successfully generated {len(dataset)} perfectly formatted agent scenarios to {output_path}")

if __name__ == "__main__":
    main()
