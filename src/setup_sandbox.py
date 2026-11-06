import os
import subprocess

def setup_sandbox():
    container_name = "ai-sandbox"
    image_name = "python:3.10-slim"
    workspace_dir = os.path.abspath(os.getcwd())

    print(f"Stopping existing container '{container_name}' (if any)...")
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

    print(f"Starting new container '{container_name}'...")
    cmd = [
        "docker", "run", "-d",
        "--name", container_name,
        "--memory=512m",
        "--cpus=1.0",
        "--network=none", # disable internet access for safety
        "-v", f"{workspace_dir}:/workspace",
        "-w", "/workspace",
        image_name,
        "tail", "-f", "/dev/null"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("Sandbox started successfully!")
        print(f"Container ID: {result.stdout.strip()}")
    else:
        print("Failed to start sandbox.")
        print(result.stderr)

if __name__ == "__main__":
    setup_sandbox()
