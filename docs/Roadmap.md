# AI Coding Assistant — Development Roadmap

This document outlines the 9-phase development roadmap for building the self-hosted AI software-engineering agent.

## Phase 1: LLM Fundamentals
**Goal:** Understand the underlying technologies of Large Language Models.
- [x] Understand Transformers architecture
- [x] Tokenization strategies
- [x] Attention mechanisms
- [x] Embeddings and latent space
- [x] Model weights and parameters
- [x] Inference pipelines
- [x] Quantization techniques (FP16, INT8, INT4)

---

## Phase 2: Local Coding Model
**Goal:** Run a small coding model locally to establish a baseline.
- [x] Select a 1B–3B parameter coding model (e.g., Qwen2.5-Coder-1.5B, DeepSeek-Coder-1.3B)
- [x] Set up local inference environment
- [x] Build a simple coding chatbot interface

---

## Phase 3: Fine-Tuning Pipeline
**Goal:** Establish the training pipeline on local hardware (RTX 4050 6GB).
- [x] Prepare an initial small dataset
- [x] Implement Supervised Fine-Tuning (SFT)
- [x] Implement LoRA (Low-Rank Adaptation)
- [x] Implement QLoRA for VRAM-constrained training
- [x] Establish evaluation metrics

---

## Phase 4: Repository Intelligence
**Goal:** Give the agent the ability to understand codebases without stuffing the context window.
- [x] Integrate Tree-sitter for AST parsing
- [x] Integrate Language Server Protocol (LSP)
- [x] Build code indexing capabilities
- [x] Implement symbol search (functions, classes, variables)
- [x] Build context retrieval mechanisms (RAG with Qdrant/pgvector)

---

## Phase 5: The Agent Loop
**Goal:** Transform the LLM from a chatbot into an autonomous agent.
- [x] Implement the Planner, Coder, and Reviewer modules
- [x] Build the Tool System framework
- [x] Implement File operations tools (read, write, edit)
- [x] Implement Terminal execution tools
- [x] Implement Git integration tools
- [x] Implement Test runner integration
- [x] Build the core Agent Loop (Task -> Plan -> Act -> Observe -> Finish)

---

## Phase 6: Secure Sandbox
**Goal:** Ensure safe execution of LLM-generated code.
- [x] Set up Docker-based execution environment
- [x] Implement CPU and RAM resource limits
- [x] Implement network restrictions
- [x] Set execution timeouts
- [x] Implement a strict permission system

---

## Phase 7: Agentic Dataset Generation
**Goal:** Create high-quality data to teach the model software engineering behavior.
- [x] Collect/generate repository-level tasks
- [x] Create bug fix scenarios
- [x] Create pull request review scenarios
- [x] Create test failure resolution scenarios
- [x] Create refactoring tasks
- [x] Log and format terminal interactions and tool usage

---

## Phase 8: Advanced Training
**Goal:** Improve the model's reasoning and agentic capabilities using advanced techniques.
- [x] Experiment with Continued Pretraining
- [x] Advanced SFT with the agentic dataset
- [x] Investigate Direct Preference Optimization (DPO)
- [x] Investigate Reinforcement Learning from Human Feedback (RLHF)
- [x] Investigate Group Relative Policy Optimization (GRPO)
- [x] Implement Executable Environment Training (training based on code execution success)

---

## Phase 9: Scale Model
**Goal:** Move to larger, more capable models once the pipeline and architecture are proven.
*Note: Only scale when evaluation demonstrates that additional model capability is necessary.*
- [x] Transition from 1B-3B to 7B parameter models (Requires cloud GPUs)
- [x] Transition to 14B parameter models
- [x] Transition to 30B+ parameter models
- [x] Investigate Large Mixture of Experts (MoE) architectures
