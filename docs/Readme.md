# AI Coding Assistant — Model & Training Strategy

## 1. Project Vision

Build a self-hosted AI coding assistant inspired by modern AI coding IDEs such as Antigravity, Cursor, and Claude Code.

The goal is not simply to create a chatbot that generates code.

The system should eventually be capable of:

- Understanding complete repositories
- Planning software changes
- Reading and modifying files
- Searching code
- Running terminal commands
- Running tests
- Debugging errors
- Refactoring code
- Using Git
- Understanding project architecture
- Iteratively fixing problems
- Working autonomously through multi-step coding tasks

The long-term objective is to develop a highly capable coding model combined with an agentic software-engineering system.

---

## 2. Core Principle

The LLM is only the **brain** of the coding assistant.

The complete system consists of:

```text
AI Coding IDE
│
├── LLM
├── Agent Runtime
├── Context Engine
├── Repository Intelligence
├── RAG
├── Tool System
├── Code Execution
├── Sandbox
├── Testing
├── Memory
└── IDE Interface
```

The final product should therefore be treated as an **AI coding agent platform**, not merely a fine-tuned LLM.

---

# 3. Model Strategy

The project will use an existing open-weight coding model rather than training a foundation model from scratch.

Possible model families:

- Qwen Coder
- Qwen3-Coder
- DeepSeek Coder
- Other strong open-weight coding models available during development

The model selection will depend on:

- Coding capability
- Agentic capability
- Context length
- License
- Model size
- VRAM requirements
- Fine-tuning support
- Tool-calling capabilities
- Inference performance

---

# 4. Do NOT Train From Scratch Initially

Training an LLM from scratch requires:

- Massive datasets
- Large GPU clusters
- Distributed training
- Significant engineering
- Large storage infrastructure
- Extremely high compute cost

Therefore:

```text
Train from scratch
        ↓
        ❌
```

Instead:

```text
Existing open-weight model
        ↓
Fine-tuning
        ↓
Specialized coding/agent model
```

---

# 5. Fine-Tuning Strategy

The project will initially investigate:

- Supervised Fine-Tuning (SFT)
- LoRA
- QLoRA
- Parameter-Efficient Fine-Tuning (PEFT)

The pretrained model's original weights can remain frozen while small trainable adapter parameters are optimized.

Conceptually:

```text
Pretrained Model
       │
       ├── Original weights
       │      ↓
       │    Frozen
       │
       └── LoRA Adapter
              ↓
          Trainable
```

The resulting system becomes:

```text
Base Model + LoRA Adapter
```

---

# 6. Quantization

Quantization reduces the numerical precision used to represent model weights.

Common formats include:

```text
FP32
FP16
BF16
INT8
INT4
```

Approximate weight memory:

```text
7B model

FP16  ≈ 14 GB
INT8  ≈ 7 GB
INT4  ≈ 3.5 GB
```

These are approximate weight-only calculations. Actual VRAM requirements are higher because inference and training also require memory for activations, KV cache, runtime buffers, and other tensors.

---

# 7. Full Precision vs Quantization

## High-Precision Model

Advantages:

- Maximum numerical precision
- Better suited to high-quality training
- More GPU memory required
- More expensive inference

Typical formats:

```text
BF16
FP16
```

## Quantized Model

Advantages:

- Much lower memory usage
- Lower VRAM requirements
- Faster/cheaper inference in many environments
- Allows larger models to run on smaller hardware

Possible formats:

```text
INT8
INT4
```

Quantization itself does not make a model more intelligent.

It primarily changes the efficiency of storing and executing the model.

---

# 8. Project Preference

The long-term goal is to optimize for **model quality first**.

Therefore:

```text
Training
    ↓
High-quality model
    ↓
Best possible checkpoint
    ↓
Deployment optimization
    ↓
Quantized versions when required
```

The project should not permanently restrict the model to the capabilities of the current development laptop.

---

# 9. Current Hardware

Development machine:

```text
CPU  : Intel Core i7-14700HX
RAM  : 16 GB
GPU  : NVIDIA RTX 4050 Laptop GPU
VRAM : 6 GB
```

This machine is suitable for:

- Model experimentation
- Small model inference
- Dataset preparation
- Agent development
- RAG development
- Repository indexing
- Small LoRA/QLoRA experiments
- Local evaluation
- Backend development
- IDE development

It is NOT intended for:

- Training large foundation models
- Full fine-tuning large models
- Training 30B+ models
- Training large MoE models
- Large-scale distributed training

---

# 10. Local Model Strategy

Initial experiments should use smaller coding models.

Recommended progression:

```text
1B
 ↓
1.5B
 ↓
3B
 ↓
7B
```

The first objective is to understand the complete pipeline rather than immediately training the largest possible model.

---

# 11. QLoRA Strategy for Current Hardware

Because the current GPU has only 6 GB VRAM:

```text
Base Model
      ↓
4-bit Quantization
      ↓
Frozen Quantized Model
      ↓
LoRA Adapter
      ↓
Train Adapter
```

This is the primary local experimentation strategy.

Small models should be used first.

7B QLoRA experiments may be possible with highly constrained settings, but should not be considered the primary local target.

---

# 12. Future GPU Strategy

Serious training will use cloud or dedicated GPU infrastructure.

Potential progression:

```text
Local RTX 4050
       ↓
Small experiments
       ↓
24 GB GPU
       ↓
7B-class experiments
       ↓
48/80 GB GPUs
       ↓
14B/30B-class experiments
       ↓
Multi-GPU infrastructure
       ↓
Large MoE models
```

The exact GPU requirements must be calculated from:

- Model size
- Precision
- Quantization
- Sequence length
- Batch size
- Gradient accumulation
- LoRA configuration
- Optimizer
- Training method
- Number of GPUs

---

# 13. Important VRAM Concept

Model parameter count does not equal total VRAM requirement.

Training memory can include:

```text
Model Weights
+
Gradients
+
Optimizer States
+
Activations
+
KV Cache / Context Memory
+
CUDA Runtime
+
Temporary Buffers
```

Therefore:

```text
7B × 0.5 bytes
```

is only a rough calculation for INT4 weight storage.

It does not represent the complete GPU requirement.

---

# 14. AI Coding Assistant Architecture

The target architecture:

```text
                    AI CODING IDE
                          │
                          ▼
                    Agent Router
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
       Planner          Coder          Reviewer
          │               │               │
          └───────────────┼───────────────┘
                          ▼
                     LLM Gateway
                          │
                    Coding Model
                          │
                          ▼
                   Context Engine
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
      Repository        RAG            Memory
      Intelligence
          │               │               │
          └───────────────┼───────────────┘
                          ▼
                         Tools
                          │
       ┌──────────┬───────┼───────┬──────────┐
       ▼          ▼       ▼       ▼          ▼
     Files     Terminal   Git    Search     Tests
       │          │       │       │          │
       └──────────┴───────┼───────┴──────────┘
                          ▼
                       Sandbox
                          │
                          ▼
                      Execution
                          │
                          ▼
                       Results
                          │
                          └──────→ Agent
```

---

# 15. Agent Loop

The coding agent should operate iteratively.

Example:

```text
User Request
     ↓
Understand task
     ↓
Create plan
     ↓
Inspect repository
     ↓
Find relevant files
     ↓
Read code
     ↓
Modify files
     ↓
Run tests
     ↓
Analyze errors
     ↓
Fix errors
     ↓
Run tests again
     ↓
Success
     ↓
Explain changes
```

This loop is fundamental to the system.

---

# 16. Training Objective

The goal is NOT simply to teach the model programming syntax.

The base coding model already has substantial programming knowledge.

The fine-tuning objective should focus heavily on:

### Repository Understanding

```text
Task
 ↓
Repository inspection
 ↓
Relevant files
 ↓
Architecture understanding
```

### Planning

```text
Requirement
 ↓
Implementation plan
 ↓
Ordered actions
```

### Code Modification

```text
Existing code
+
Requirement
 ↓
Minimal safe modification
```

### Debugging

```text
Code
+
Error
 ↓
Root cause
 ↓
Fix
```

### Testing

```text
Modification
 ↓
Run tests
 ↓
Analyze results
 ↓
Correct implementation
```

### Agentic Behavior

```text
Task
 ↓
Tool selection
 ↓
Tool execution
 ↓
Observe result
 ↓
Next action
```

---

# 17. Training Dataset

The dataset should eventually contain high-quality software-engineering tasks.

Potential categories:

```text
dataset/
│
├── code_generation/
├── debugging/
├── refactoring/
├── code_review/
├── repository_understanding/
├── test_generation/
├── terminal_tasks/
├── git_tasks/
├── dependency_management/
├── architecture_tasks/
└── agentic_tasks/
```

A high-quality example may contain:

```text
Task
+
Repository state
+
Available tools
+
Agent actions
+
Tool results
+
Code changes
+
Test results
+
Final solution
```

---

# 18. Executable Training

A major long-term goal is to train against executable environments.

Instead of:

```text
Question → Answer
```

use:

```text
Task
 ↓
Agent Action
 ↓
Execute
 ↓
Environment Feedback
 ↓
Agent Action
 ↓
Execute
 ↓
Test
 ↓
Success / Failure
```

This allows the model to learn software-engineering behavior through actual environment feedback.

---

# 19. Repository Intelligence

The system should not send an entire repository into the model context.

Instead:

```text
Repository
    ↓
Tree-sitter / AST
    ↓
Symbols
    ↓
Functions
    ↓
Classes
    ↓
Dependencies
    ↓
Index
    ↓
Context Retrieval
```

This allows the system to retrieve only relevant code.

---

# 20. RAG

RAG should handle information that does not need to be permanently encoded into model weights.

Potential sources:

- Project documentation
- API documentation
- Framework documentation
- Company documentation
- Repository knowledge
- User-provided documents

Possible stack:

```text
Embeddings
    ↓
Qdrant / pgvector
    ↓
Retriever
    ↓
Context Builder
    ↓
LLM
```

---

# 21. Tool System

The model should request tools instead of directly controlling the operating system.

Example:

```text
LLM
 ↓
Tool Request
 ↓
Permission / Policy Layer
 ↓
Sandbox
 ↓
Tool Execution
 ↓
Result
 ↓
LLM
```

Potential tools:

```text
read_file
write_file
edit_file
search_code
search_files
run_terminal
run_tests
git_diff
git_status
git_commit
browser_search
documentation_search
```

---

# 22. Security

The coding agent will eventually execute arbitrary code.

Therefore a sandbox is mandatory.

Possible technologies:

```text
Docker
Resource limits
Filesystem isolation
Network restrictions
Process limits
Execution timeout
Permission system
```

The LLM must never have unrestricted direct access to the host operating system.

---

# 23. Technology Stack

## Model

```text
Qwen Coder
Qwen3-Coder
DeepSeek Coder
Other open-weight coding models
```

## Training

```text
Python
PyTorch
Hugging Face Transformers
Hugging Face Datasets
PEFT
TRL
Accelerate
bitsandbytes
FlashAttention
```

## Fine-Tuning

```text
SFT
LoRA
QLoRA
PEFT
```

## Inference

```text
vLLM
```

## Backend

```text
FastAPI
Python
```

## Repository Intelligence

```text
Tree-sitter
LSP
Git
```

## RAG

```text
Qdrant
or
PostgreSQL + pgvector
```

## Database

```text
PostgreSQL
```

## Cache

```text
Redis
```

## Sandbox

```text
Docker
```

## IDE

Possible approaches:

```text
VS Code Extension
Electron
Tauri
```

---

# 24. Development Roadmap

## Phase 1 — LLM Fundamentals

Learn:

- Transformers
- Tokenization
- Attention
- Embeddings
- Weights
- Inference
- Quantization

---

## Phase 2 — Local Coding Model

Run a small coding model locally.

Target:

```text
1B–3B
```

Build:

```text
Simple coding chatbot
```

---

## Phase 3 — Fine-Tuning

Implement:

```text
Dataset
 ↓
SFT
 ↓
LoRA
 ↓
QLoRA
 ↓
Evaluation
```

---

## Phase 4 — Repository Intelligence

Implement:

```text
Tree-sitter
LSP
Code indexing
Symbol search
Context retrieval
```

---

## Phase 5 — Agent

Implement:

```text
Planner
Tool system
File operations
Terminal
Git
Test runner
Agent loop
```

---

## Phase 6 — Sandbox

Implement secure:

```text
Docker execution
Resource limits
Network restrictions
Timeouts
Permissions
```

---

## Phase 7 — Agentic Dataset

Collect/generate:

```text
Repository tasks
Bug fixes
Pull requests
Test failures
Refactoring tasks
Terminal interactions
```

---

## Phase 8 — Advanced Training

Investigate:

```text
Continued Pretraining
SFT
DPO
RLHF
GRPO
Agentic Training
Executable Environment Training
```

---

## Phase 9 — Scale Model

Move from:

```text
1B–3B
   ↓
7B
   ↓
14B
   ↓
30B+
   ↓
Large MoE
```

only when evaluation demonstrates that additional model capability is necessary.

---

# 25. Final Long-Term Architecture

The final vision:

```text
                    YOUR AI CODING IDE
                           │
                           ▼
                     Agent Runtime
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
          Planner        Coder        Reviewer
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                    Model Gateway
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
        Large Coding Model       Fast Small Model
              │                         │
              └────────────┬────────────┘
                           ▼
                     Context Engine
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   AST / LSP              RAG               Memory
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                         Tools
                           │
        ┌──────────┬───────┼───────┬──────────┐
        ▼          ▼       ▼       ▼          ▼
      Files     Terminal   Git    Search     Tests
        │          │       │       │          │
        └──────────┴───────┼───────┴──────────┘
                           ▼
                        Sandbox
                           │
                           ▼
                       Execution
                           │
                           ▼
                      Environment
                           │
                           └────────→ Agent Loop
```

---

# 26. Core Philosophy

The project should follow these principles:

1. **Do not train from scratch initially.**
2. **Start with a strong open-weight coding model.**
3. **Use LoRA/QLoRA for early experiments.**
4. **Use the current RTX 4050 for development and small experiments.**
5. **Use cloud GPUs for serious training.**
6. **Optimize model quality before deployment compression.**
7. **Focus fine-tuning on agentic software-engineering behavior.**
8. **Use RAG for external and changing knowledge.**
9. **Use AST/LSP-based repository intelligence.**
10. **Use executable environments for agent evaluation and future training.**
11. **Never allow unrestricted LLM access to the host OS.**
12. **Evaluate the system on real repository-level coding tasks.**
13. **Scale the model only when evaluation proves that a larger model is needed.**

---

# 27. Current Project Position

Current hardware:

```text
RTX 4050 6GB
16GB RAM
i7-14700HX
```

Current target:

```text
Small coding model
+
QLoRA
+
Agent architecture
```

Future target:

```text
Large open-weight coding model
+
High-quality agentic training
+
RAG
+
Repository intelligence
+
Tool execution
+
Sandbox
+
Automated testing
```

Ultimate objective:

> Build a highly capable, self-hosted AI software-engineering agent that can understand, modify, execute, test, debug, and improve real-world software repositories with minimal human intervention.