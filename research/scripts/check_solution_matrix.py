#!/usr/bin/env python3
"""Check solution directory coverage against models/problems configuration.

The script constructs the expected solution grid as
``len(models) * len(problems) * len(variant_indices)`` (where
``variant_indices`` are read from ``num_solutions.txt``) and compares it
against the contents of ``solutions/``. It also validates that ``research/``
matches the entries listed in ``problems.txt``.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


MODEL_PREFIX_ALIASES: Dict[str, str] = {
    "grokcodefast1_": "grok4fastreasoning_",
}


def read_list_file(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    items: List[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        items.append(line)
    if not items:
        raise ValueError(f"No valid entries found in {path}")
    return items


def read_problem_list(path: Path) -> List[str]:
    problems = []
    for entry in read_list_file(path):
        normalized = entry.split("research/", 1)[-1]
        problems.append(normalized)
    return problems


def read_models_list(path: Path) -> List[str]:
    models: List[str] = []
    seen: set[str] = set()
    for entry in read_list_file(path):
        if entry not in seen:
            models.append(entry)
            seen.add(entry)
    return models


def read_variant_indices(path: Path) -> List[int]:
    values = read_list_file(path)
    if len(values) == 1:
        try:
            n = int(values[0])
            if n <= 0:
                return [0]
            return list(range(n))
        except ValueError:
            pass
    indices: List[int] = []
    seen: set[int] = set()
    for v in values:
        try:
            idx = int(v)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid variant index in {path}: '{v}'") from exc
        if idx < 0:
            raise ValueError(f"Variant indices must be >= 0, got {idx}")
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)
    if not indices:
        return [0]
    return indices


def canonical_model_prefix(model: str) -> str:
    original = model
    if "/" in model:
        model = model.split("/", 1)[1]
    model_lower = model.lower().strip()

    # Keep GPT-5.1 distinct from GPT-5 to avoid collapsing
    # directories/solution names.
    if model_lower.startswith("gpt-5.1") or model_lower.startswith("gpt5.1"):
        return "gpt5.1"
    if model_lower.startswith("gpt-5") or model_lower.startswith("gpt5"):
        return "gpt5"

    if "gemini-2.5-pro" in model_lower or "gemini2.5pro" in model_lower:
        return "gemini2.5pro"

    gemini_match = re.match(r"gemini-?(\d+\.?\d*)-?pro", model_lower)
    if gemini_match:
        version = gemini_match.group(1)
        return f"gemini{version}pro"

    claude_match = re.match(r"claude-([a-z]+)-(\d+)-(\d+)", model_lower)
    if claude_match:
        family = claude_match.group(1)
        major = claude_match.group(2)
        minor = claude_match.group(3)
        return f"claude{major}.{minor}{family}"

    sanitized = re.sub(r"[^a-zA-Z0-9]+", "", model_lower)
    if "grok" in model_lower and "fast" in model_lower and sanitized:
        return sanitized
    if not sanitized:
        raise ValueError(f"Unable to derive model prefix from '{original}'")
    return sanitized


def sanitize_problem_name(problem: str) -> str:
    return problem.replace("/", "_")


def expected_solution_names(
    problems: Iterable[str],
    model_prefixes: Sequence[str],
    variant_indices: Sequence[int],
) -> Tuple[Dict[Tuple[str, str], List[str]], Dict[str, str]]:
    mapping: Dict[Tuple[str, str], List[str]] = {}
    slug_to_problem: Dict[str, str] = {}

    for problem in problems:
        slug = sanitize_problem_name(problem)
        slug_to_problem[slug] = problem
        for prefix in model_prefixes:
            names: List[str] = []
            base = f"{prefix}_{slug}"
            for idx in (list(variant_indices) or [0]):
                if idx == 0:
                    names.append(base)
                else:
                    names.append(f"{base}_{idx}")
            mapping[(prefix, problem)] = names
    return mapping, slug_to_problem


def collect_solution_dirs(root: Path) -> List[str]:
    if not root.is_dir():
        return []
    return sorted(entry.name for entry in root.iterdir() if entry.is_dir())


def collect_problem_dirs(root: Path) -> List[str]:
    candidates: set[str] = set()
    if not root.is_dir():
        return []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if len(relative.parts) > 2:
            continue
        entries = {child.name.lower() for child in path.iterdir() if child.is_file()}
        if any(name in entries for name in {"readme.md", "readme"}):
            candidates.add(str(relative))
    return sorted(candidates)


def summarize_matrix(
    problems_file: Path,
    models_file: Path,
    num_solutions_file: Path,
    solutions_dir: Path,
    problems_dir: Path,
    pairs_output: Path,
) -> None:
    problems = read_problem_list(problems_file)
    models = read_models_list(models_file)
    variant_indices = read_variant_indices(num_solutions_file)

    if not problems:
        raise SystemExit("No problems configured.")
    if not models:
        raise SystemExit("No models configured.")

    model_prefixes = [canonical_model_prefix(model) for model in models]

    expected_map, slug_to_problem = expected_solution_names(problems, model_prefixes, variant_indices)
    expected_dirs = {name for names in expected_map.values() for name in names}

    pair_lines = [
        f"{solution}:{problem}"
        for (_, problem), solutions in sorted(expected_map.items(), key=lambda item: (item[0][1], item[0][0]))
        for solution in sorted(solutions)
    ]
    header = "# auto-generated pairs from scripts/check_solution_matrix.py\n"
    pairs_output.write_text(header + "\n".join(pair_lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(pair_lines)} pairs to {pairs_output}")

    actual_dirs = set(collect_solution_dirs(solutions_dir))
    normalized_dirs = set()
    for name in actual_dirs:
        remapped = False
        for old_prefix, new_prefix in MODEL_PREFIX_ALIASES.items():
            if name.startswith(old_prefix):
                normalized_dirs.add(new_prefix + name[len(old_prefix):])
                remapped = True
                break
        if not remapped:
            normalized_dirs.add(name)
    actual_dirs = normalized_dirs

    missing_by_problem: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for key, names in expected_map.items():
        model_prefix, problem = key
        for name in names:
            if name not in actual_dirs:
                missing_by_problem[key].append(name)

    extra_dirs = sorted(actual_dirs - expected_dirs)

    missing_problem_dirs = [p for p in problems if not (problems_dir / p).exists()]

    actual_problem_dirs = collect_problem_dirs(problems_dir)
    extra_problem_dirs = sorted(set(actual_problem_dirs) - set(problems))

    print("=== Solution Matrix Check ===")
    print(f"Problems listed: {len(problems)}")
    print(f"Models listed:   {len(models)} ({', '.join(model_prefixes)})")
    print(
        f"Generations per model/problem: {len(variant_indices)} (indices: {', '.join(map(str, variant_indices))})"
    )
    print(f"Expected solution directories: {len(expected_dirs)}")
    print(f"Actual solution directories:   {len(actual_dirs)}")
    print("")

    total_missing = sum(len(names) for names in missing_by_problem.values())
    print(f"Missing solution directories: {total_missing}")
    if total_missing:
        for (model_prefix, problem), names in sorted(missing_by_problem.items()):
            gens_missing = [name for name in names]
            print(f"  - {model_prefix} -> {problem}: missing {len(gens_missing)}")
            for entry in gens_missing[:5]:
                print(f"      · {entry}")
            if len(gens_missing) > 5:
                print(f"      · ... ({len(gens_missing) - 5} more)")
    print("")

    print(f"Extra solution directories: {len(extra_dirs)}")
    if extra_dirs:
        for name in extra_dirs[:20]:
            model_hint = name.split("_", 1)[0] if "_" in name else name
            slug = name.split("_", 1)[1] if "_" in name else ""
            hint = ""
            if slug:
                problem_hint = slug_to_problem.get(slug)
                if problem_hint:
                    hint = f" (possible problem: {problem_hint})"
            print(f"  - {name}{hint}")
        if len(extra_dirs) > 20:
            print(f"  - ... ({len(extra_dirs) - 20} more)")
    print("")

    print("=== Problems Directory Check ===")
    print(f"Missing problem directories: {len(missing_problem_dirs)}")
    for path in missing_problem_dirs[:10]:
        print(f"  - {path}")
    if len(missing_problem_dirs) > 10:
        print(f"  - ... ({len(missing_problem_dirs) - 10} more)")

    print(f"Extra problem directories: {len(extra_problem_dirs)}")
    for path in extra_problem_dirs[:10]:
        print(f"  - {path}")
    if len(extra_problem_dirs) > 10:
        print(f"  - ... ({len(extra_problem_dirs) - 10} more)")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify solution/problem coverage.")
    parser.add_argument("--problems", default="research/problems.txt", help="Path to problems.txt (default: problems.txt)")
    parser.add_argument("--models", default="research/models.txt", help="Path to models.txt (default: models.txt)")
    parser.add_argument(
        "--num-solutions",
        default="research/num_solutions.txt",
        help=(
            "Path to num_solutions.txt listing variant indices per line "
            "(0 means no suffix). Backward-compatible with a single integer N."
        ),
    )
    parser.add_argument("--solutions-dir", default="solutions", help="Solutions directory (default: solutions)")
    parser.add_argument("--problems-dir", default="problems", help="Problems directory (default: problems)")
    parser.add_argument(
        "--pairs-output",
        default="all_pairs.txt",
        help="Write derived solution/problem pairs to this file (default: all_pairs.txt)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summarize_matrix(
            Path(args.problems),
            Path(args.models),
            Path(args.num_solutions),
            Path(args.solutions_dir),
            Path(args.problems_dir),
            Path(args.pairs_output),
        )
    except Exception as exc:  # pragma: no cover - CLI feedback
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
