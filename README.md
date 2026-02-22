<h1 align="center">
  <img src="https://raw.githubusercontent.com/AI-at-Scale/book-companion/main/images/kdp_front_cover.jpg" alt="AI at Scale Cover" width="300" />
  <br>
  AI at Scale: Companion Code Repository
</h1>

<h4 align="center">The official code, playbooks, and calculators for <strong>AI at Scale: The Physics and Playbooks of Trillion-Parameter Systems</strong>.</h4>

<p align="center">
  <a href="https://github.com/AI-at-Scale/book-companion/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  </a>
  <a href="https://github.com/AI-at-Scale/book-companion/issues">
    <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome">
  </a>
  <a href="https://github.com/AI-at-Scale/book-companion/network/members">
    <img src="https://img.shields.io/github/forks/AI-at-Scale/book-companion?style=social" alt="Forks">
  </a>
  <a href="https://github.com/AI-at-Scale/book-companion/stargazers">
    <img src="https://img.shields.io/github/stars/AI-at-Scale/book-companion?style=social" alt="Stars">
  </a>
</p>

<p align="center">
  <a href="#-about-the-book">About The Book</a> •
  <a href="#-reading-paths-by-role">Reading Paths</a> •
  <a href="#-the-killer-tools">The "Killer Tools"</a> •
  <a href="#-installation--setup">Installation</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## 📖 About the Book

Building AI that scales isn't just about throwing more GPUs at the problem. It requires a fundamental understanding of the physics of computing, the bottlenecks of data transfer, and the economics of infrastructure. This repository provides the practical, "copy-paste-and-run" resources discussed in the book to help you bridge the gap between theory and immediate execution.

## 🧭 Reading Paths by Role

To get the most out of this repository, we recommend following the path that matches your role:

### 🏢 For Architects (CTO / VP Engineering)
*Focus: Budgeting, GreenOps, and Governance.*
- **Start Here:** [`tools/scaling_law_tco_calculator.py`](tools/scaling_law_tco_calculator.py)
- **Key Chapters:** `chapter_02_data_refinery/`, `war_room_simulations/`, `chapter_07_agents/`
- **Goal:** Justify infrastructure spend and forecast model performance before writing a check.

### ⚙️ For Operators (Senior ML Engineer)
*Focus: Distributed Training, Parallelism, and Debugging.*
- **Start Here:** [`tools/parallelism_planner.py`](tools/parallelism_planner.py)
- **Key Chapters:** `war_room_simulations/`, `chapter_03_parallelism/`
- **Goal:** Architect a hybrid 3D parallelism strategy to avoid OOM crashes and maximize cluster utilization.

### 🔨 For Builders (ML Infrastructure Lead)
*Focus: Inference Optimization, Serving, and Latency.*
- **Start Here:** [`tools/inference_cost_planner.py`](tools/inference_cost_planner.py)
- **Key Chapters:** `chapter_05_inference/`, `war_room_simulations/`
- **Goal:** Optimize KV cache, implement continuous batching, and halve your inference bill.

## 🛠️ The "Killer Tools"

We have included three interactive calculators in the `tools/` directory to help you plan your architecture:

| Tool | Path | Description |
| ---- | ---- | ----------- |
| **Scaling Law TCO Calculator** | [`tools/scaling_law_tco_calculator.py`](tools/scaling_law_tco_calculator.py) | Predicts optimal parameter ($N$) and token ($D$) counts based on your hardware budget. |
| **Parallelism Planner** | [`tools/parallelism_planner.py`](tools/parallelism_planner.py) | Generates a recommended DP/TP/PP/CP sharding strategy based on your model size and cluster topology. |
| **Inference Cost Planner** | [`tools/inference_cost_planner.py`](tools/inference_cost_planner.py) | Calculates KV cache sizes, continuous batching memory budgets, and Total Cost of Ownership (TCO) per token. |

## 🚀 Installation & Setup

All scripts are written in standard Python 3.10+. Some scripts rely on `torch` and `numpy`.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AI-at-Scale/book-companion.git
   cd book-companion
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install numpy torch
   ```

4. **Run a tool (Example):**
   ```bash
   python tools/scaling_law_tco_calculator.py
   ```

## 🤝 Contributing

We welcome contributions! Have you built an extension to one of the tools, or found a typo in a code snippet? Please read our [Contributing Guidelines](CONTRIBUTING.md) to see how you can help.

---

<p align="center">
  <i>If you find these resources helpful, please consider leaving a ⭐ on the repository and reviewing the book on Amazon.</i>
</p>
