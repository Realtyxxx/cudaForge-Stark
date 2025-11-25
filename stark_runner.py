from __future__ import annotations
import argparse
import random
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents.query_server import query_server
from prompts.generate_custom_cuda import build_seed_prompt, default_system_prompt
from prompts.stark import build_plan_prompt, build_code_prompt, build_debug_prompt
from scripts.individual import KernelIndividual
from utils.kernel_io import extract_code_block, extract_json, save_kernel_code

# Reuse core helpers from the baseline runner
from main import (  # noqa: E402
    _bench_and_score,
    _build_run_tag,
    _build_arg_parser,
    _append_usage_totals,
    _collect_tasks,
    _last_n_lines,
    _pick_first_n,
    _plot_scores,
    _sample_tasks,
    _save_global_summary,
)


@dataclass
class StarkNode:
    node_id: int
    parent_id: Optional[int]
    code: str
    code_path: Path
    metrics: Optional[Dict[str, Any]] = None
    score: float = float("-inf")
    plan_json: Optional[str] = None
    anchored_code: Optional[str] = None
    plan_raw: Optional[str] = None
    logs: str = ""
    phase: str = ""
    created_round: int = 0
    children: List[int] = field(default_factory=list)

    @property
    def runnable(self) -> bool:
        return bool(self.metrics and self.metrics.get("runnable", False))

    @property
    def buggy(self) -> bool:
        return self.metrics is not None and not self.metrics.get("runnable", False)

    def header(self) -> str:
        base = f"node={self.node_id} parent={self.parent_id} score={self.score:.4f}"
        status = "runnable" if self.runnable else "buggy"
        return f"{base} [{status}]"


class StarkTree:
    def __init__(self, epsilon: float = 0.25, root_limit: int = 4, leaf_bias: float = 1.5, leader_topk: int = 3, rng: Optional[random.Random] = None):
        self.nodes: Dict[int, StarkNode] = {}
        self.root_id: Optional[int] = None
        self.next_id: int = 0
        self.epsilon = epsilon
        self.root_limit = root_limit
        self.leaf_bias = leaf_bias
        self.leader_topk = leader_topk
        self.leader_ids: List[int] = []
        self.rng = rng or random.Random()

    def add_node(self, node: StarkNode) -> None:
        self.nodes[node.node_id] = node
        if self.root_id is None:
            self.root_id = node.node_id
        if node.parent_id is not None and node.parent_id in self.nodes:
            self.nodes[node.parent_id].children.append(node.node_id)

        if node.runnable and node.score is not None:
            self.leader_ids.append(node.node_id)
            self.leader_ids = sorted(self.leader_ids, key=lambda nid: self.nodes[nid].score or float("-inf"), reverse=True)[
                : self.leader_topk
            ]

    def _candidate_filter(self, node: StarkNode) -> bool:
        if node.node_id == self.root_id and len(node.children) >= self.root_limit:
            return False
        return True

    def _leaves(self, nodes: List[StarkNode]) -> List[StarkNode]:
        return [n for n in nodes if not n.children]

    def select_node(self) -> Optional[StarkNode]:
        if not self.nodes:
            return None

        runnable = [n for n in self.nodes.values() if n.runnable and self._candidate_filter(n)]
        buggy = [n for n in self.nodes.values() if n.buggy and self._candidate_filter(n)]
        pool = runnable or buggy or list(self.nodes.values())

        if not pool:
            return None

        explore = self.rng.random() < self.epsilon
        if explore:
            leaves = self._leaves(pool) or pool
            weights = [self.leaf_bias if not n.children else 1.0 for n in leaves]
            return self.rng.choices(leaves, weights=weights, k=1)[0]

        # exploitation: prefer best leader, else best score
        for lid in self.leader_ids:
            n = self.nodes[lid]
            if self._candidate_filter(n):
                return n

        return max(pool, key=lambda n: n.score or float("-inf"))


def _call_agent(prompt: str, system_prompt: str, *, args, temperature: float, log_path: Optional[Path] = None, call_type: str = "unknown", round_idx: int = -1) -> str:
    return query_server(
        prompt=prompt,
        system_prompt=system_prompt,
        server_type=args.server_type,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        temperature=temperature,
        top_p=args.top_p,
        server_address=args.server_address,
        server_port=args.server_port,
        log_path=str(log_path) if log_path else None,
        call_type=call_type,
        round_idx=round_idx,
    )


def _format_nodes(nodes: List[StarkNode], title: str, limit: int = 3) -> str:
    if not nodes:
        return f"## {title}\n(none)\n"
    lines: List[str] = [f"## {title} (showing {min(limit, len(nodes))})"]
    for n in nodes[:limit]:
        lines.append(f"- {n.header()}")
        lines.append("```python")
        lines.append(n.code.strip())
        lines.append("```")
    return "\n".join(lines) + "\n"


def _summaries_for_plan(tree: StarkTree, node: StarkNode) -> Tuple[str, str]:
    # Local context: children + cousins (siblings' children)
    children = [tree.nodes[cid] for cid in node.children]
    cousins: List[int] = []
    if node.parent_id is not None and node.parent_id in tree.nodes:
        parent = tree.nodes[node.parent_id]
        for sib_id in parent.children:
            if sib_id == node.node_id:
                continue
            cousins.extend(tree.nodes[sib_id].children)
    cousin_nodes = [tree.nodes[c] for c in cousins]
    local_context = _format_nodes(children + cousin_nodes, "Local history", limit=4)
    leader_nodes = [tree.nodes[lid] for lid in tree.leader_ids]
    leader_context = _format_nodes(leader_nodes, "Global leaders", limit=3)
    return local_context, leader_context


def _context_for_code(tree: StarkTree, node: StarkNode) -> str:
    siblings: List[StarkNode] = []
    if node.parent_id is not None and node.parent_id in tree.nodes:
        parent = tree.nodes[node.parent_id]
        siblings = [tree.nodes[sid] for sid in parent.children if sid != node.node_id]
    cousin_children: List[StarkNode] = []
    for sib in siblings:
        cousin_children.extend(tree.nodes[cid] for cid in sib.children)
    leaders = [tree.nodes[lid] for lid in tree.leader_ids if lid != node.node_id]
    return _format_nodes(siblings + cousin_children + leaders, "Neighbours & leaders", limit=5)


def _context_for_debug(tree: StarkTree, node: StarkNode) -> str:
    siblings: List[StarkNode] = []
    if node.parent_id is not None and node.parent_id in tree.nodes:
        parent = tree.nodes[node.parent_id]
        siblings = [tree.nodes[sid] for sid in parent.children if sid != node.node_id]
    leaders = [tree.nodes[lid] for lid in tree.leader_ids if lid != node.node_id]
    return _format_nodes(siblings + leaders, "Siblings/Leaders", limit=4)


def _make_individual(code: str, code_dir: Path) -> KernelIndividual:
    ind = KernelIndividual(code)
    path = save_kernel_code(code, code_dir)
    ind.code_path = path  # type: ignore[attr-defined]
    return ind


def _run_stark_task(task_path: Path, args, batch_dir: Path, tree_params: Dict[str, Any]) -> Dict[str, Any]:
    task_root = (batch_dir / task_path.stem).resolve()
    code_dir = task_root / "code"
    eval_dir = task_root / "evaluation"
    fig_dir = task_root / "figures"
    io_dir = eval_dir / "llm_io"

    code_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    io_dir.mkdir(parents=True, exist_ok=True)
    log_path = task_root / "usage.csv"

    # Write reference model to stable path for benchmarking
    root_dir = Path(__file__).resolve().parent
    ref_py = root_dir / f"ref_{args.subproc_id}.py"
    ref_py.write_text(task_path.read_text(encoding="utf-8"), encoding="utf-8")

    rng = random.Random(args.shuffle_seed or None)
    tree = StarkTree(
        epsilon=tree_params["epsilon"],
        root_limit=tree_params["root_limit"],
        leaf_bias=tree_params["leaf_bias"],
        leader_topk=tree_params["leader_topk"],
        rng=rng,
    )

    scores: List[float] = []
    err_flags: List[bool] = []
    best_node: Optional[StarkNode] = None
    last_score_for_curve = 0.0 # change the default to zero

    # -------- Seed node --------
    seed_prompt = build_seed_prompt(arch_path=task_path, gpu_name=args.gpu)
    (io_dir / "round000_seed_prompt.txt").write_text(seed_prompt, encoding="utf-8")
    raw_seed = _call_agent(seed_prompt, default_system_prompt, args=args, temperature=args.plan_temperature, log_path=log_path, call_type="seed", round_idx=0)
    (io_dir / "round000_seed_reply.txt").write_text(raw_seed, encoding="utf-8")
    seed_code = extract_code_block(raw_seed) or raw_seed
    # seed_code = open("/home/tanyanxi/workspace/fake_stark/chatgpt/kernel2_14.py", "r").read()
    seed_ind = _make_individual(seed_code, code_dir)
    _bench_and_score(
        seed_ind,
        ref_py=task_path,
        device_idx=args.device,
        warmup=args.warmup,
        repeat=args.repeat,
        tol=args.tol,
        phase="seed",
        metrics_dir=eval_dir,
        timeout_s=args.bench_timeout,
    )
    seed_node = StarkNode(
        node_id=tree.next_id,
        parent_id=None,
        code=seed_ind.code,
        code_path=seed_ind.code_path,  # type: ignore[arg-type]
        metrics=getattr(seed_ind, "metrics", None),
        score=seed_ind.score or float("-inf"),
        phase="seed",
        created_round=0,
        logs=_last_n_lines(getattr(seed_ind, "metrics", {}).get("message", ""),
                           120) if getattr(seed_ind, "metrics", None) else "",
    )
    tree.next_id += 1
    tree.add_node(seed_node)
    best_node = seed_node if seed_node.runnable else None
    if seed_node.runnable:
        last_score_for_curve = seed_node.score
        scores.append(seed_node.score)
        err_flags.append(False)
    else:
        scores.append(last_score_for_curve)
        err_flags.append(True)

    # -------- Expansion loop --------
    for round_idx in range(1, args.round):
        node = tree.select_node()
        if node is None:
            print("[STARK] No selectable node; stopping.")
            break

        print(f"[STARK] Round {round_idx} expanding {node.header()}")
        plan_raw: Optional[str] = None
        plan_json_str: str = ""
        anchored_code: Optional[str] = None
        # --- Choose branch ---
        if not node.runnable:
            # Debug branch
            dbg_context = _context_for_debug(tree, node)
            plan_hint = node.plan_json or ""
            dbg_sys, dbg_prompt = build_debug_prompt(
                error_log=_last_n_lines(node.logs, 120),
                current_kernel=node.code,
                context=dbg_context,
                plan_hint=plan_hint,
            )
            (io_dir / f"round{round_idx:03d}_debug_prompt.txt").write_text(dbg_prompt, encoding="utf-8")
            raw = _call_agent(dbg_prompt, dbg_sys, args=args, temperature=args.debug_temperature, log_path=log_path, call_type="debug", round_idx=round_idx)
            (io_dir / f"round{round_idx:03d}_debug_reply.txt").write_text(raw, encoding="utf-8")
            new_code = extract_code_block(raw) or raw
            phase = "debug"
            plan_json_str = plan_hint
            anchored_code = None
        else:
            # Optimisation branch: Plan -> Code
            local_ctx, leader_ctx = _summaries_for_plan(tree, node)
            plan_sys, plan_prompt = build_plan_prompt(node.code, local_ctx, leader_ctx)
            (io_dir / f"round{round_idx:03d}_plan_prompt.txt").write_text(plan_prompt, encoding="utf-8")
            plan_raw = _call_agent(plan_prompt, plan_sys, args=args, temperature=args.plan_temperature, log_path=log_path, call_type="plan", round_idx=round_idx)
            (io_dir / f"round{round_idx:03d}_plan_reply.txt").write_text(plan_raw, encoding="utf-8")
            try:
                plan_json_obj = extract_json(plan_raw)
                if isinstance(plan_json_obj, (dict, list)):
                    plan_json_str = json.dumps(plan_json_obj, ensure_ascii=False, indent=2)
                else:
                    plan_json_str = str(plan_json_obj)
            except Exception:
                plan_json_str = ""

            anchored_code = extract_code_block(plan_raw) or node.code
            code_context = _context_for_code(tree, node)
            code_sys, code_prompt = build_code_prompt(plan_json_str or "(no plan)", anchored_code, code_context)
            (io_dir / f"round{round_idx:03d}_code_prompt.txt").write_text(code_prompt, encoding="utf-8")
            code_raw = _call_agent(code_prompt, code_sys, args=args, temperature=args.code_temperature, log_path=log_path, call_type="code", round_idx=round_idx)
            (io_dir / f"round{round_idx:03d}_code_reply.txt").write_text(code_raw, encoding="utf-8")
            new_code = extract_code_block(code_raw) or code_raw
            phase = "opt"

        ind = _make_individual(new_code, code_dir)
        _bench_and_score(
            ind,
            ref_py=task_path,
            device_idx=args.device,
            warmup=args.warmup,
            repeat=args.repeat,
            tol=args.tol,
            phase=phase,
            metrics_dir=eval_dir,
            timeout_s=args.bench_timeout,
        )
        new_node = StarkNode(
            node_id=tree.next_id,
            parent_id=node.node_id,
            code=ind.code,
            code_path=ind.code_path,  # type: ignore[arg-type]
            metrics=getattr(ind, "metrics", None),
            score=ind.score or float("-inf"),
            plan_json=plan_json_str if node.runnable else plan_hint,
            anchored_code=anchored_code,
            plan_raw=plan_raw if node.runnable else None,
            phase=phase,
            created_round=round_idx,
            logs=_last_n_lines(getattr(ind, "metrics", {}).get("message", ""),
                               120) if getattr(ind, "metrics", None) else "",
        )
        tree.next_id += 1
        tree.add_node(new_node)

        runnable = new_node.runnable
        if runnable and (best_node is None or (new_node.score or float("-inf")) > (best_node.score or float("-inf"))):
            best_node = new_node
        if runnable and new_node.score is not None:
            last_score_for_curve = new_node.score
            scores.append(new_node.score)
            err_flags.append(False)
        else:
            scores.append(last_score_for_curve)
            err_flags.append(True)

    # plot per-task curve
    fig_path = fig_dir / f"{task_path.stem}_stark_score.png"
    _plot_scores(fig_path, scores, err_flags,
                 title=f"{task_path.stem} (best={best_node.score if best_node else float('-inf'):.4f})")
    print(f"[STARK] Figure saved to: {fig_path}")

    # Save best kernel for convenience
    if best_node:
        best_path = task_root / "best_kernel.py"
        best_path.write_text(best_node.code, encoding="utf-8")

    usage_totals = _append_usage_totals(log_path)

    return {
        "task": str(task_path),
        "best_score": float(best_node.score) if best_node and best_node.score is not None else 0.0,
        "best_runnable": bool(best_node.runnable) if best_node else False,
        "task_dir": str(task_root),
        "figure": str(fig_path),
        "input_tokens_sum": usage_totals["input_tokens"],
        "output_tokens_sum": usage_totals["output_tokens"],
        "total_tokens_sum": usage_totals["total_tokens"],
    }


def _build_stark_arg_parser() -> argparse.ArgumentParser:
    p = _build_arg_parser()
    p.description = "STARK-style multi-agent CUDA kernel search"
    p.add_argument("--epsilon", type=float, default=0.25, help="epsilon-greedy exploration rate for tree search")
    p.add_argument("--leader_topk", type=int, default=3, help="number of top leaders to keep in context windows")
    p.add_argument("--root_limit", type=int, default=4,
                   help="max children allowed for root before throttling selection")
    p.add_argument("--leaf_bias", type=float, default=1.5, help="sampling weight boost for leaves during exploration")
    p.add_argument("--plan_temperature", type=float, default=0.8, help="Plan agent temperature")
    p.add_argument("--code_temperature", type=float, default=0.1, help="Code agent temperature")
    p.add_argument("--debug_temperature", type=float, default=0.1, help="Debug agent temperature")
    return p


def stark_main():
    args = _build_stark_arg_parser().parse_args()
    all_tasks = _collect_tasks(args.arch_py)

    # Build batch folder name
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = _build_run_tag(args.server_type, args.model_name)
    if args.arch_py.is_file():
        batch_name = f"{ts}_{args.arch_py.stem}_stark_{run_tag}"
    else:
        pick_note = f"first{args.first_n}" if (args.first_n and args.first_n >
                                               0) else f"num{args.num_tasks}_seed{args.shuffle_seed}"
        batch_name = f"{ts}_batch_{pick_note}_stark_{run_tag}"
    batch_dir = (args.work_dir / batch_name).resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"[STARK] Output folder: {batch_dir}")

    tree_params = {
        "epsilon": args.epsilon,  # NOTE: Exploration Rate
        "root_limit": args.root_limit,  # NOTE: Root Node Limit
        "leaf_bias": args.leaf_bias,  # NOTE: The Propability to add leaf node when in exploration
        "leader_topk": args.leader_topk,  # NOTE: Leader Top K
    }

    summary: List[Dict[str, Any]] = []

    if args.arch_py.is_file():
        res = _run_stark_task(all_tasks[0], args, batch_dir, tree_params)
        summary.append(res)
    else:
        if args.first_n and args.first_n > 0:
            picked = _pick_first_n(all_tasks, args.first_n)
            print(f"[STARK] Found {len(all_tasks)} tasks, taking first {len(picked)} (sorted).")
        else:
            picked = _sample_tasks(all_tasks, args.num_tasks, args.shuffle_seed)
            print(f"[STARK] Found {len(all_tasks)} tasks, sampled {len(picked)} with seed={args.shuffle_seed}.")

        for i, task in enumerate(picked, 1):
            print(f"\n===== [STARK {i}/{len(picked)}] Running task: {task} =====")
            res = _run_stark_task(task, args, batch_dir, tree_params)
            summary.append(res)

    if summary:
        avg_speedup = sum(s["best_score"] for s in summary) / len(summary)
        accuracy = sum(1 for s in summary if s["best_runnable"]) / len(summary)
        total_tokens_sum = sum(int(s.get("total_tokens_sum", 0) or 0) for s in summary)
        _save_global_summary(batch_dir, summary, avg_speedup, accuracy, total_tokens_sum)
        print(f"[STARK] Avg speedup={avg_speedup:.4f}, Accuracy={accuracy:.4f}")
    else:
        print("[STARK] No tasks were run.")


if __name__ == "__main__":
    stark_main()
