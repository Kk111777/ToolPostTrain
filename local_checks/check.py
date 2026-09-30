"""Run exact upstream pure functions without importing the CUDA/Ray training stack.

AST selection changes no function bodies. It deliberately does not verify package
imports, distributed aggregation, rollouts, or model training.
"""

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import platform
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "verl-GDPO/verl/trainer/ppo/core_algos.py"
FUNCTIONAL = ROOT / "verl-GDPO/verl/utils/torch_functional.py"
REWARD = ROOT / "verl-GDPO/verl/utils/reward_score/rlla.py"


def load_upstream_functions():
    namespace = {"torch": torch, "defaultdict": defaultdict}
    for path, names in [
        (CORE, {"compute_grpo_outcome_advantage"}),
        (FUNCTIONAL, {"masked_mean", "masked_var", "masked_whiten"}),
    ]:
        tree = ast.parse(path.read_text())
        nodes = [
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        if {n.name for n in nodes} != names:
            raise RuntimeError(f"Upstream function set changed: {path}")
        exec(  # noqa: S102 - execute selected, unmodified functions from the pinned local upstream
            compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
            namespace,
        )
    return namespace


FUNCTIONS = load_upstream_functions()


def official_advantages(rewards, mask, groups, method):
    """Use the two-reward branch from upstream ray_trainer.compute_advantage."""
    token_rewards = []
    for column in range(rewards.shape[1]):
        reward = torch.zeros_like(mask)
        last_tokens = mask.sum(dim=1).long() - 1
        if (last_tokens < 0).any():
            raise ValueError("Each response needs at least one valid token")
        reward[torch.arange(len(rewards)), last_tokens] = rewards[:, column]
        token_rewards.append(reward)
    grpo = FUNCTIONS["compute_grpo_outcome_advantage"]
    if method == "grpo":
        return grpo(sum(token_rewards), mask, groups)[0]
    if method != "gdpo":
        raise ValueError(method)
    normalized = sum(grpo(reward, mask, groups)[0] for reward in token_rewards)
    return FUNCTIONS["masked_whiten"](normalized, mask) * mask


def independent_numpy_reference(rewards, mask, groups, method):
    """Independent vectorized reference for the selected VERL convention."""
    scores = rewards.sum(axis=1, keepdims=True) if method == "grpo" else rewards.copy()
    centered = np.empty_like(scores)
    for group in np.unique(groups):
        selected = np.asarray(groups) == group
        values = scores[selected]
        if len(values) == 1:
            centered[selected] = values / (1.0 + 1e-6)
        else:
            centered[selected] = (values - values.mean(axis=0)) / (
                values.std(axis=0, ddof=1) + 1e-6
            )
    result = centered.sum(axis=1)[:, None] * mask
    if method == "gdpo":
        valid = result[mask.astype(bool)]
        result = (result - valid.mean()) / np.sqrt(valid.var(ddof=1) + 1e-8) * mask
    return result


def check_numerics():
    rng = np.random.default_rng(20260920)
    errors = {}
    fixtures = {
        "mixed_lengths": rng.normal(size=(8, 2)).astype(np.float32),
        "zero_variance": np.zeros((8, 2), dtype=np.float32),
        "constant_one_reward": np.column_stack([np.ones(8), rng.normal(size=8)]).astype(
            np.float32
        ),
    }
    groups = ["p0"] * 4 + ["p1"] * 4
    mask = (np.arange(5)[None, :] < np.array([1, 2, 3, 4, 5, 2, 4, 3])[:, None]).astype(
        np.float32
    )
    for fixture, rewards in fixtures.items():
        for method in ("grpo", "gdpo"):
            actual = official_advantages(
                torch.tensor(rewards), torch.tensor(mask), groups, method
            ).numpy()
            expected = independent_numpy_reference(rewards, mask, groups, method)
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
            assert np.isfinite(actual).all()
            assert np.all(actual[mask == 0] == 0)
            errors[f"{fixture}/{method}"] = float(np.abs(actual - expected).max())
    return errors


def check_rewards():
    # Import the actual upstream reward file; no reimplementation of its scorer.
    spec = importlib.util.spec_from_file_location("upstream_rlla", REWARD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    truth = '<think>fixture</think>\n<tool_call>\n{"name":"get_stock","parameters":{"sku":"A"}}\n</tool_call>'
    fixtures = {
        "correct": (truth, (4.0, 1.0, 3.0, 0.0)),
        "wrong_tool": (
            truth.replace("get_stock", "wrong_tool"),
            (-2.0, 1.0, -3.0, 0.0),
        ),
        "malformed": ("not a tool call", (-3.0, 0.0, -3.0, 0.0)),
    }
    overrides = {
        key: "0"
        for key in (
            "WITHLENGTH",
            "REFINEDREWARD",
            "COARSEREWARD",
            "STRICTMATCH",
            "CORRECTMAX1",
            "INTERMEDIATEREWARD",
            "MAX1STEP30MAX3",
            "SCHEDULEREWARD",
            "SCHEDULELENGTH",
        )
    }
    overrides["EXPERIMENT_NAME"] = "qwen-local-check"
    old = {key: os.environ.get(key) for key in overrides}
    scores = {}
    try:
        os.environ.update(overrides)
        for name, (completion, expected) in fixtures.items():
            with contextlib.redirect_stdout(io.StringIO()):
                score = module.compute_score(completion, truth)
            np.testing.assert_allclose(score, expected)
            scores[name] = score
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return scores


def main():
    report = {
        "scope": "upstream_pure_function_cpu_validation_not_training",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "upstream_commit": subprocess.check_output(
            ["git", "rev-parse", "upstream/main"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (CORE, FUNCTIONAL, REWARD)
        },
        "numerical_max_abs_error": check_numerics(),
        "reward_fixtures": check_rewards(),
        "model_parameters_updated": False,
        "passed": True,
    }
    output = ROOT / "artifacts/local-checks.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
