from __future__ import annotations
"""Prompt builder for Mind‑Evolution CUDA‑kernel search (seed‑kernel version).

Generates a **single prompt** that contains:
1. Target GPU spec (from `prompts/hardware/gpu_specs.py`)
2. **Few‑shot pair** – original *and* optimised model code blocks
3. Source architecture (`class Model`) that needs to be optimised
4. Existing kernel summaries (optional, for diversity context)
5. A **diversity requirement** section ensuring the new kernel differs from all previous ones
6. Output requirements

CLI usage
---------
```bash
python -m prompts.build_prompt KernelBench/level1/19_ReLU.py \
       --gpu "Quadro RTX 6000" -o prompt.txt
```
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from string import Template
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]  # project root
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"  # GPU spec table
PROMPTS_DIR = Path(__file__).parent / "stark_prompts" / "custom_cuda"

# --------------------------------------------------
# Few‑shot pair  (before / after)
# --------------------------------------------------
FEWSHOT_BASE = ROOT / "prompts/few_shot/model_ex_add.py"   # original Model
FEWSHOT_NEW = ROOT / "prompts/few_shot/model_new_ex_add.py"  # optimised ModelNew

# ---------------------------------------------------------------------------
# Load Prompts from files
# ---------------------------------------------------------------------------

# Version configuration
TEST_TEMPLATE_VERSION = "v1"
MAIN_TEMPLATE_VERSION = "v1"
SYSTEM_PROMPT_VERSION = "v1"

def load_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()

test = Template(load_prompt(f"test_template_{TEST_TEMPLATE_VERSION}.txt"))
TEMPLATE = Template(load_prompt(f"main_template_{MAIN_TEMPLATE_VERSION}.txt"))
default_system_prompt = load_prompt(f"system_{SYSTEM_PROMPT_VERSION}.txt")

# ---------------------------------------------------------------------------
# GPU spec loader
# ---------------------------------------------------------------------------


def _load_gpu_spec() -> dict:  # noqa: D401
    """Import `gpu_specs.py` and return the GPU_SPEC_INFO dict (robust across Python versions)."""
    spec = importlib.util.spec_from_file_location("gpu_specs", HW_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {HW_FILE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["gpu_specs"] = module  # avoid re‑import
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    if not hasattr(module, "GPU_SPEC_INFO"):
        raise AttributeError("GPU_SPEC_INFO not defined in gpu_specs.py")
    return module.GPU_SPEC_INFO  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Prompt builder core
# ---------------------------------------------------------------------------

def build_seed_prompt(
    arch_path: Path,
    gpu_name: str | None = None,
) -> str:
    """Build LLM prompt for CUDA‑kernel optimisation (seed generation)."""
    gpu_info = _load_gpu_spec()

    # Auto‑detect GPU if not provided
    if gpu_name is None:
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("CUDA device not found – pass --gpu <name>.") from exc

    if gpu_name not in gpu_info:
        raise KeyError(f"{gpu_name} not present in GPU_SPEC_INFO")

    info = gpu_info[gpu_name]
    gpu_arch = info.get("GPU Architecture", "Unknown")
    gpu_items = "\n".join(
        f"• {k}: {v}" for k, v in info.items() if k != "GPU Architecture"
    )

    few_base = FEWSHOT_BASE.read_text().strip()
    few_new = FEWSHOT_NEW.read_text().strip()
    arch_src = Path(arch_path).read_text().strip()

    return test.substitute(
        few_base=few_base,
        few_new=few_new,
        # arch_src=arch_src,
        arch_src=gpu_items,
        kernel_src=arch_src
    )


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def _cli() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(
        description="Build LLM prompt for CUDA‑kernel optimisation (seed generation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model_py", help="Path to .py containing class Model")
    parser.add_argument("--gpu", default=None, help="GPU name key in gpu_specs.py")
    parser.add_argument("-o", "--out", help="Save prompt to file")
    args = parser.parse_args()

    # Fixed: was calling undefined build_prompt
    prompt = build_seed_prompt(Path(args.model_py), args.gpu)

    if args.out:
        Path(args.out).write_text(prompt)
        print(f"[✓] Prompt saved to {args.out}")
    else:
        print(prompt)


if __name__ == "__main__":  # pragma: no cover
    _cli()
