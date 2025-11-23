from __future__ import annotations
from string import Template
from pathlib import Path

# Configuration for prompt versions
PLAN_VERSION = "v1"
CODE_VERSION = "v1"
DEBUG_VERSION = "v1"

PROMPTS_DIR = Path(__file__).parent / "stark_prompts"

def load_prompt_content(category: str, filename: str) -> str:
    path = PROMPTS_DIR / category / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()

# Load Plan Prompts
PLAN_SYSTEM_PROMPT = load_prompt_content("plan", f"system_{PLAN_VERSION}.txt")
PLAN_PROMPT_TMPL = Template(load_prompt_content("plan", f"user_template_{PLAN_VERSION}.txt"))

# Load Code Prompts
CODE_SYSTEM_PROMPT = load_prompt_content("code", f"system_{CODE_VERSION}.txt")
CODE_PROMPT_TMPL = Template(load_prompt_content("code", f"user_template_{CODE_VERSION}.txt"))

# Load Debug Prompts
DEBUG_SYSTEM_PROMPT = load_prompt_content("debug", f"system_{DEBUG_VERSION}.txt")
DEBUG_PROMPT_TMPL = Template(load_prompt_content("debug", f"user_template_{DEBUG_VERSION}.txt"))


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
