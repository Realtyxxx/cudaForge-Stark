# STARK Paper Memory Prompts

**Source**: [STARK Paper Note](main-note.md) | [PDF](../../../Library/Mobile%20Documents/com~apple~CloudDocs/papers/stark/2510.16996v1.pdf)

## Core Concept
**STARK (Strategic Team of Agents for Refining Kernels)** is a multi-agent LLM framework for optimizing GPU kernels. It mimics an expert engineering team by separating concerns into **Plan**, **Code**, and **Debug** agents, and uses **Strategic Search** over a tree of attempts instead of linear refinement.

## Key Components
- **Multi-Agent Collaboration**:
  - **Plan Agent** ($\tau=0.8$): Proposes optimization strategies (tiling, vectorization) and inserts **Grounded Instructions** (anchors like `<<<IMPROVE>>>`) into code.
  - **Code Agent** ($\tau=0.1$): Implements the plan within the anchored regions.
  - **Debug Agent** ($\tau=0.1$): Fixes compilation/runtime errors based on logs.
- **Strategic Search**: Uses an adapted $\epsilon$-greedy policy on a search tree to balance exploration and exploitation, avoiding local optima.
- **Dynamic Context Window**: Tailors context for each agent (e.g., Plan agent sees global leaders; Code agent sees similar "cousin" implementations).

## Algorithm Summary
1. **Initialize**: Search tree with root (PyTorch source).
2. **Loop**:
   - **Select**: Pick a node using $\epsilon$-greedy.
   - **Branch**:
     - If buggy -> **Debug** (fix errors).
     - If valid -> **Optimize** (Plan -> Code).
   - **Evaluate**: Compile & measure runtime.
   - **Update**: Add new node to tree & update leaderboard.
3. **Output**: Best kernel from leaderboard.

## Key Results (KernelBench)
- **Performance**: Up to **16x speedup** over baseline agents (Reflexion, Sampling).
- **Success Rate**: **100%** on Level 1 & 2 tasks; significantly outperforms baselines on hard Level 3 tasks.
- **Efficiency**: High correctness rate (61.2% on Level 2) due to structured planning.
