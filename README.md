<div align="center">

# 🤖 Fine-Tune LLM
### Self-Hosted AI Coding Agent — From QLoRA Fine-Tuning to Autonomous Software Engineering

[![CI](https://github.com/nfsprogramming/Fine-Tune-LLM/actions/workflows/ci.yml/badge.svg)](https://github.com/nfsprogramming/Fine-Tune-LLM/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

</div>

---

## 📖 Overview

**Fine-Tune LLM** is an open-source research and engineering toolkit for training, evaluating, and running a **self-hosted AI coding agent** on consumer hardware.

Rather than relying on closed cloud APIs or basic conversational chatbots, this project implements a complete local software-engineering agent platform:
- Fine-tunes compact coding LLMs (e.g., `Qwen2.5-Coder-1.5B-Instruct`) using **4-bit QLoRA** within a 6 GB VRAM envelope.
- Implements a resilient **ReAct (Reasoning + Acting)** execution loop with multi-format tool call parsing.
- Provides isolated, resource-constrained **Docker sandbox** execution for terminal and test commands.
- Indexes codebase repositories using **BGE embeddings** and a local **Qdrant** vector store for semantic code retrieval (RAG).
- Merges trained LoRA adapters into standalone models ready for offline inference.

---

## 🏗️ System Architecture

```
                       User Prompt / Task
                              │
                              ▼
                    ┌───────────────────┐
                    │    Agent Loop     │◄───────────────────┐
                    │   (src/agent.py)  │                    │
                    └─────────┬─────────┘                    │
                              │                              │
                     Chat Template + Prompt                  │
                              │                              │
                              ▼                              │
                    ┌───────────────────┐                    │
                    │  Fine-Tuned LLM   │                    │
                    │   (Qwen2.5-Coder) │                    │
                    └─────────┬─────────┘                    │
                              │                              │
                     Model Generation                        │
                    (<thought> + Action)                     │
                              │                              │
                              ▼                              │
                    ┌───────────────────┐                    │
                    │ Multi-Format Tool │                    │
                    │    Call Parser    │                    │
                    └─────────┬─────────┘                    │
                              │                              │
           ┌──────────────────┴──────────────────┐           │
           ▼                                     ▼           │
 ┌───────────────────┐                 ┌───────────────────┐ │
 │ File System Tools │                 │  Sandbox Engine   │ │
 │ (read/write file) │                 │  (Docker isolated)│ │
 └─────────┬─────────┘                 └─────────┬─────────┘ │
           │                                     │           │
           └──────────────────┬──────────────────┘           │
                              │                              │
                       Tool Observation                      │
                              └──────────────────────────────┘
```

---

## ✨ Features

- **Consumer-Grade QLoRA Fine-Tuning**: Supervised fine-tuning using 4-bit NormalFloat (`nf4`) quantization with `bitsandbytes`, `peft`, and `trl`, engineered to train on GPUs with 6 GB VRAM.
- **Autonomous Agent Loop**: Bounded multi-step reasoning loop (`<thought>` + action + observation) executing file operations, terminal commands, and git workflows.
- **Fault-Tolerant Tool Call Parser**: Intercepts five distinct tool call formats:
  1. Standard XML tags (`<tool_call>{"name": ..., "arguments": ...}</tool_call>`).
  2. Markdown file write headers (``path/file.ext: ```...``` ``).
  3. Fenced JSON code blocks (`````json {...} `````).
  4. Qwen native function calls (`FunctionName\n{...}`).
  5. Fallback raw code block extraction.
- **Docker Execution Sandbox**: Dedicated lightweight Linux sandbox (`python:3.10-slim`) constrained to 512 MB RAM, 1 CPU core, and disabled network access for safe code execution.
- **Repository Intelligence & RAG**: Semantic chunking and vector indexing of codebases via `BAAI/bge-small-en-v1.5` embeddings stored in a local embedded `Qdrant` vector database.
- **Context-Augmented Chatbot**: Interactive terminal interface combining vector search retrieval with streaming model responses.
- **Dataset Synthesis & Filtering**: Tools to generate structured synthetic agent scenarios and filter public tool-calling datasets (`pyromind/agentic-tool-call-dataset-12k`).
- **Comprehensive Unit Tests & CI**: Automated testing of parser heuristics, tool execution, and dataset generators with GitHub Actions matrix verification.

---

## 📁 Project Structure

```
Fine-Tune-LLM/
├── .github/
│   └── workflows/
│       └── ci.yml              # Multi-version Python CI pipeline
├── adapter/                    # Exported LoRA adapter metadata & weights
│   ├── adapter_config.json
│   └── README.md
├── data/                       # Seed and generated training datasets
│   ├── agentic_dataset.jsonl   # Synthetic tool-calling dataset
│   └── dataset.jsonl           # Curated software-engineering SFT examples
├── docs/                       # Project research documentation
│   ├── Cloud_Scaling_Guide.md  # Scaling from 1.5B to 7B/14B/32B on cloud GPUs
│   ├── Conclusion.md           # Architecture summary and findings
│   ├── Readme.md               # Detailed model training strategy
│   └── Roadmap.md              # 9-phase development roadmap
├── scripts/                    # Automation and setup utilities
│   ├── download_coder_model.py # Download Qwen2.5-Coder model variants
│   ├── download_dataset.py     # Download and filter Hugging Face agentic datasets
│   ├── download_model.py       # Download BGE embedding model
│   └── generate_dataset.py     # Generate synthetic agentic training data
├── src/                        # Core application source code
│   ├── agent.py                # Autonomous agent loop & tool parsing
│   ├── chatbot.py              # Interactive RAG chatbot with streaming
│   ├── evaluate.py             # LoRA model evaluation script
│   ├── indexer.py              # Qdrant code chunker and vector indexer
│   ├── merge_model.py          # Merge LoRA adapter weights into base model
│   ├── setup_sandbox.py        # Docker sandbox container provisioning
│   ├── tools.py                # Tool definitions (filesystem, terminal, git)
│   └── train.py                # 4-bit QLoRA SFT training pipeline
├── tests/                      # Automated test suite
│   ├── test_agent_parser.py    # Unit tests for multi-syntax tool parsing
│   ├── test_dataset_generator.py # Unit tests for dataset generation
│   └── test_tools.py           # Unit tests for tool executions and prompts
├── .gitignore                  # Git exclusion rules for models and caches
├── index.html                  # Sample web interface artifact
├── LICENSE                     # MIT License
├── pyproject.toml              # Project metadata and dependencies specification
└── requirements.txt            # Locked pip dependencies
```

---

## 🧰 Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Base Models** | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Base reasoning and code generation foundation |
| **Fine-Tuning** | `trl` (`SFTTrainer`), `peft` (LoRA), `accelerate` | Quantized parameter-efficient fine-tuning |
| **Quantization** | `bitsandbytes` | 4-bit NormalFloat (`nf4`) quantization |
| **Embeddings** | `BAAI/bge-small-en-v1.5` | Dense vector representations for codebase search |
| **Vector DB** | `qdrant-client` | Embedded local vector database storage |
| **Sandbox** | Docker (`python:3.10-slim`) | Resource-constrained command execution |
| **Runtime** | Python 3.10+, PyTorch (CUDA / CPU) | Core language and tensor computation backend |

---

## 💻 Hardware Requirements

The training pipeline is optimized for entry-level consumer GPUs:

| Hardware Component | Baseline Specification | Target Workload |
| :--- | :--- | :--- |
| **GPU** | NVIDIA RTX 4050 Laptop (6 GB VRAM) | 4-bit QLoRA SFT on 1.5B parameter models |
| **System RAM** | 16 GB | Model loading and dataset batching |
| **Storage** | ~20 GB free space | Model checkpoints and local vector database |
| **Cloud Scaling** | 16 GB–80 GB VRAM (RTX 4090 / A100) | 7B, 14B, and 32B model fine-tuning (see [`docs/Cloud_Scaling_Guide.md`](docs/Cloud_Scaling_Guide.md)) |

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/nfsprogramming/Fine-Tune-LLM.git
cd Fine-Tune-LLM
```

### 2. Set Up the Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download the Base Model and Embeddings

```bash
# Download base Qwen2.5-Coder-1.5B model
python scripts/download_coder_model.py --size 1.5b

# Download BGE embedding model for code search
python scripts/download_model.py
```

### 4. Prepare or Generate Training Datasets

```bash
# Generate synthetic tool-calling dataset:
python scripts/generate_dataset.py

# Or download and filter Hugging Face agentic datasets:
python scripts/download_dataset.py
```

### 5. Fine-Tune with QLoRA

Train the LoRA adapter on your GPU:

```bash
python src/train.py
```

Checkpoints are saved to `results/` and final adapter weights to `adapter/`.

### 6. Merge Adapter Weights

Merge the trained LoRA adapter directly into the base model weights:

```bash
python src/merge_model.py
```

The resulting standalone model is saved to `models/Qwen2.5-Coder-1.5B-Agent`.

---

## 🛠️ Usage

### Indexing a Codebase for Semantic Search (RAG)

Before querying the chatbot or agent about a project, index the codebase into Qdrant:

```bash
python src/indexer.py
```

This scans eligible source files (`.py`, `.js`, `.ts`, `.md`, `.jsonl`, `.txt`), generates embeddings, and saves the vector store in `./qdrant_db`.

### Running the Interactive RAG Chatbot

```bash
python src/chatbot.py
```

Ask questions about your indexed codebase. Context is dynamically retrieved from Qdrant and passed to the model.

### Running the Autonomous Agent

1. Start the Docker sandbox container:
   ```bash
   python src/setup_sandbox.py
   ```
2. Launch the agent:
   ```bash
   python src/agent.py
   ```
3. Enter software-engineering tasks in the prompt:
   ```text
   Task: Inspect src/tools.py, find where test_runner is defined, and write a new test in tests/test_demo.py
   ```

---

## 🧪 Testing

Run the automated unit test suite using Python's standard `unittest` runner:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

To run individual test modules:
```bash
python -m unittest tests/test_tools.py
python -m unittest tests/test_agent_parser.py
python -m unittest tests/test_dataset_generator.py
```

---

## ⚠️ Known Limitations

- **VRAM Constraints**: 4-bit QLoRA is configured for 1.5B parameter models on 6 GB VRAM. Training 7B+ models requires minimum 16 GB VRAM.
- **Sandbox Dependency**: Terminal and test execution tools require Docker to be installed and running. If Docker is unavailable, filesystem tools (`read_file`, `write_file`) continue to operate natively.
- **Context Length**: The default training configuration uses a 1024-token sequence length to prevent CUDA out-of-memory errors on small GPUs.

---

## 🤝 Contributing

Contributions, bug reports, and discussions are welcome:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/agent-enhancement`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Verify all tests pass (`python -m unittest discover -s tests`).
5. Push to the branch and open a Pull Request.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
