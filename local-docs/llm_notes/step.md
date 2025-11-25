# Worklog: STARK Integration

## Plan
- Understand STARK requirements and pipeline integration points.
- Implement STARK multi-agent/search components (prompts, tree, agents) and hook into runner/CLI.
- Smoke-check code paths and document usage.

## Implemented
- Added `prompts/stark.py`: Plan/Code/Debug agent prompts with grounded anchors and API-preserving rules.
- Added `stark_runner.py`: epsilon-greedy tree search (root throttling, leaf bias, leader top-k), dynamic contexts for each agent, Plan→Code→Debug branching, and reuse of existing benchmarking/saving pipeline.
- Updated `README.md`: usage example and flag descriptions for the experimental STARK runner.
- Basic syntax check via `python -m py_compile stark_runner.py prompts/stark.py`.

## Summary / Notes
- New entrypoint: `python3 stark_runner.py <task_or_dir> --gpu "<GPU name>" --round <N> --epsilon 0.25 --leader_topk 3 --plan_temperature 0.8 --code_temperature 0.1 --debug_temperature 0.1 ...`
- Outputs per task under `run/<timestamp>_stark_*/` with code/evaluation/figures/llm_io and `best_kernel.py`.
- Next ideas: wire STARK runner into `main.py` CLI alias, feed Nsight metrics into plan context, and add persistence of tree/search state across runs if needed.
