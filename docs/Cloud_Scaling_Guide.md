# Phase 9: Cloud Scaling Guide

Because you developed this architecture on an RTX 4050 with 6GB of VRAM, we were limited to the 1.5B parameter models. Now that the agent loop, fine-tuning scripts, and secure sandboxing are built and verified, it is incredibly easy to scale this up to a massive 7B, 14B, or 32B parameter model!

## Step 1: Rent a Cloud GPU
You will need a GPU with more VRAM. Here are the recommendations based on the model size you want to run:
*   **7B Model:** Requires ~16GB VRAM (e.g., RTX 4080, A4000). Costs ~$0.30/hr on RunPod.
*   **14B Model:** Requires ~24GB VRAM (e.g., RTX 3090, RTX 4090). Costs ~$0.40/hr on RunPod.
*   **32B Model:** Requires ~80GB VRAM (e.g., A100 80GB). Costs ~$1.80/hr on RunPod.

**Platform Recommendations:**
1.  [RunPod](https://www.runpod.io/) (Easiest to set up, cheap community GPUs)
2.  [Lambda Labs](https://lambdalabs.com/) (Extremely cheap A100s, but often out of stock)

## Step 2: Set Up the Environment
When you spin up a cloud GPU, select a **PyTorch Template** image (RunPod provides these by default). 
SSH into your new instance and clone this repository:

```bash
git clone <your-repo-url>
cd NFS_AI_Coder
pip install -r requirements.txt
```

## Step 3: Download the Larger Model
We have updated `download_coder_model.py` to accept a size argument. Simply run it with the size you want:

```bash
python download_coder_model.py --size 7b
```

## Step 4: Run the Agent
The codebase is entirely modular. You don't need to rewrite anything! 
Change the `model_name` variable in `agent.py` to point to the new model (e.g., `Qwen/Qwen2.5-Coder-7B-Instruct`), and run your agent just like you did locally:

```bash
python agent.py
```

Because your cloud instance is running Linux, the Docker sandbox and terminal execution scripts will continue to work perfectly out-of-the-box!
