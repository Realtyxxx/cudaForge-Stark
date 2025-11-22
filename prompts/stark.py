from __future__ import annotations
from string import Template
from textwrap import dedent

PLAN_SYSTEM_PROMPT = dedent(
    """
You are the **Plan Agent** in a STARK-style multi-agent CUDA optimizer.
- Goal: propose the highest-impact optimisation and mark *where* to edit using grounded anchors.
- Style: concise, surgical, metrics-aware; avoid generic advice.
- Temperature hint: 0.8 (creative but precise).

You must produce:
1) A JSON plan summarising the main edit(s) and risks.
2) The current kernel rewritten with explicit anchors so a downstream Code Agent can fill them.
"""
)

PLAN_PROMPT_TMPL = Template(
    dedent(
        """
Context
-------
- Current kernel (node to expand):
```python
$current_kernel
```

- Local history (children / cousins, if any):
$local_context

- Global leaders (top candidates across the tree):
$leader_context

Task
----
Propose the single most promising optimisation to speed up this kernel on the target GPU.
Use **grounded instructions** by inserting anchored spans into the code. Keep untouched
regions verbatim so the Code Agent can rely on you for scope selection.

Output format (strict)
1) JSON inside ```json with keys:
   - "targets": short bullet(s) of what to change (e.g., "tile matmul, stage B in smem").
   - "tactics": the concrete optimisation method(s).
   - "risks": one line on possible regressions.
   - "expected_gain": short rationale for speedup.
2) Anchored code inside ```python using markers:
   <<<IMPROVE BEGIN:id:label>>>
   ... code to be replaced or scaffolded ...
   <<<IMPROVE END:id>>>
   Use 1-3 anchors. Do not duplicate anchors for the same region.
"""
    )
)


CODE_SYSTEM_PROMPT = dedent(
    """
You are the **Code Agent** in a STARK-style pipeline.
- Goal: realise the Plan Agent's anchored instructions into runnable CUDA extension code.
- Style: low-temperature (0.1), deterministic, keep public API identical.
- You must REMOVE all anchor markers in the final output.
"""
)

CODE_PROMPT_TMPL = Template(
    dedent(
        """
Inputs
------
- Plan (from Plan Agent):
```json
$plan_json
```

- Anchored scaffold to fill:
```python
$anchored_code
```

- Relevant neighbour kernels (for reference only):
$code_context

Task
----
Implement the plan inside the anchored regions. Preserve all other code exactly. Maintain
imports order, `source`/`cpp_src`, `load_inline`, and `class ModelNew` structure. Keep the
original inputs/outputs and tensor shapes. Remove **all** anchor markers in the final code.

Output
------
Return a single code block:
```python
# complete, runnable code with anchors resolved
```
No extra text.
"""
    )
)


DEBUG_SYSTEM_PROMPT = dedent(
    """
You are the **Debug Agent**. Fix compilation/runtime errors while keeping the optimisation intent.
- Focus on correctness first; avoid speculative rewrites.
- Use siblings/leaders as hints when uncertain.
- Output only corrected code.
"""
)

DEBUG_PROMPT_TMPL = Template(
    dedent(
        """
Error log (truncated):
$error_log

Current broken kernel:
```python
$current_kernel
```

Nearby attempts (siblings/leaders for hints):
$context

If a plan is provided, keep its intent:
$plan_hint

Return the corrected code in:
```python
# fixed code
```
"""
    )
)


def build_plan_prompt(current_kernel: str, local_context: str, leader_context: str) -> tuple[str, str]:
    user = PLAN_PROMPT_TMPL.substitute(
        current_kernel=current_kernel.strip(),
        local_context=local_context.strip() or "(no local history yet)",
        leader_context=leader_context.strip() or "(no leaders yet)",
    )
    return PLAN_SYSTEM_PROMPT, user


def build_code_prompt(plan_json: str, anchored_code: str, code_context: str) -> tuple[str, str]:
    user = CODE_PROMPT_TMPL.substitute(
        plan_json=plan_json.strip(),
        anchored_code=anchored_code.strip(),
        code_context=code_context.strip() or "(no neighbours)",
    )
    return CODE_SYSTEM_PROMPT, user


def build_debug_prompt(error_log: str, current_kernel: str, context: str, plan_hint: str = "") -> tuple[str, str]:
    user = DEBUG_PROMPT_TMPL.substitute(
        error_log=error_log.strip() or "(no log)",
        current_kernel=current_kernel.strip(),
        context=context.strip() or "(no sibling/leader context)",
        plan_hint=plan_hint.strip() or "(no plan hint)",
    )
    return DEBUG_SYSTEM_PROMPT, user
