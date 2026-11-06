import os
import subprocess
import json

def read_file(filepath):
    """Reads a file from the workspace."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {filepath}: {e}"

def write_file(filepath, content):
    """Creates or overwrites a file."""
    try:
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        return f"Error writing to file {filepath}: {e}"

def execute_terminal(command):
    """Runs a shell command securely in the docker sandbox."""
    try:
        container_name = "ai-sandbox"
        docker_cmd = [
            "docker", "exec", container_name,
            "sh", "-c", command
        ]
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[STDERR]:\n{result.stderr}"
        if not output.strip():
            return f"Command executed successfully with exit code {result.returncode} (no output)."
        return output
    except Exception as e:
        return f"Error executing command in sandbox: {e}"

def git_tool(command):
    """Runs a git command."""
    if not command.startswith("git "):
        command = "git " + command
    return execute_terminal(command)

def test_runner(command):
    """Runs a test framework command."""
    return execute_terminal(command)

AVAILABLE_TOOLS = {
    "read_file": read_file,
    "write_file": write_file,
    "execute_terminal": execute_terminal,
    "git_tool": git_tool,
    "test_runner": test_runner
}

TOOL_DESCRIPTIONS = [
    {
        "name": "read_file",
        "description": "Reads the content of a local file.",
        "parameters": {
            "filepath": "The path to the file to read."
        }
    },
    {
        "name": "write_file",
        "description": "Writes text content to a local file, overwriting it if it exists.",
        "parameters": {
            "filepath": "The path to the file.",
            "content": "The exact string content to write."
        }
    },
    {
        "name": "execute_terminal",
        "description": "Executes a shell command in the terminal and returns the output.",
        "parameters": {
            "command": "The shell command to run."
        }
    },
    {
        "name": "git_tool",
        "description": "Executes a git command in the repository.",
        "parameters": {
            "command": "The git command to run (e.g., 'status', 'log', 'add .')."
        }
    },
    {
        "name": "test_runner",
        "description": "Executes a test command.",
        "parameters": {
            "command": "The test framework command (e.g., 'pytest', 'npm test')."
        }
    }
]

def get_tool_prompt(persona="Coder"):
    prompt = f"You are an autonomous AI software engineer acting as a {persona}.\n\n"
    prompt += "You have access to the following tools to interact with the environment:\n\n"
    for tool in TOOL_DESCRIPTIONS:
        prompt += f"Tool: {tool['name']}\nDescription: {tool['description']}\nParameters: {json.dumps(tool['parameters'])}\n\n"
    prompt += "To use a tool, you MUST output a JSON block wrapped in <tool_call> tags. Example:\n"
    prompt += "<tool_call>{\"name\": \"execute_terminal\", \"arguments\": {\"command\": \"dir\"}}</tool_call>\n\n"
    prompt += "**SPECIAL INSTRUCTION FOR WRITING FILES**:\n"
    prompt += "Instead of using the JSON tool for write_file, you can simply write the filename followed by a colon and a markdown code block. Example:\n"
    prompt += "index.html:\n```html\n<h1>Hello</h1>\n```\n"
    prompt += "This will automatically be intercepted and saved to index.html.\n\n"
    prompt += "You will receive the output of your tools as an Observation. You may use as many tools as you need.\n"
    prompt += "ALWAYS think step-by-step using <thought> tags before acting.\n"
    prompt += "Once you have fully completed the user's task, output your final response."
    return prompt
