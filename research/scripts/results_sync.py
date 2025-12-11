#!/usr/bin/env python3
"""Synchronize results artifacts and provide textual analytics for Frontier-CS.

This script performs the following tasks:
    * Ensures ``results/results.csv`` mirrors the latest ``*_result.txt`` logs.
    * Reconstructs the expected solution/problem pairs from ``models.txt`` and
      ``problems.txt`` (no ``all_pairs.txt`` dependency).
    * Highlights notable score buckets (zeros, negatives, missing / N/A).

It can be used standalone or imported by other tooling (e.g. the SkyPilot helper).
"""

import argparse
import csv
import datetime as dt
import logging
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, TypeVar

try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False

RESULT_SUFFIX = "_result.txt"
SOLUTION_LINE_RE = re.compile(r"\[INFO\]\s+Preparing solution\s+(.+?)\.\.\.")
PROBLEM_LINE_RE = re.compile(r"\[INFO\]\s+Setting up problem environment for\s+(.+?)\.\.\.")
NUMERIC_LINE_RE = re.compile(r"^-?\d+(?:\.\d+)?$")

PairT = TypeVar("PairT")

MODEL_PREFIX_ALIASES: dict[str, str] = {
    "grokcodefast1_": "grok4fastreasoning_",
}


def setup_logger() -> logging.Logger:
    """Configure and return a colored logger using colorlog if available."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplication
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)

    if HAS_COLORLOG:
        # Use colorlog for colored output
        formatter = colorlog.ColoredFormatter(
            fmt='%(log_color)s[%(levelname)s]%(reset)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )
    else:
        # Fallback to plain formatter if colorlog not available
        formatter = logging.Formatter(
            fmt='[%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.propagate = False

    return logger


logger = setup_logger()


@dataclass
class Pair:
    solution: str
    problem: str


@dataclass
class ResultRecord:
    solution: str
    problem: str
    score: str
    status: str
    timestamp: str
    filename: str
    score_value: Optional[float]
    original_score_value: Optional[float]
    was_clamped: bool
    content_warnings: List[str]  # Warnings detected in result file content


# Patterns to detect potential evaluator issues in result file content
# Even if score is valid, these indicate problems worth investigating
WARNING_PATTERNS = [
    (re.compile(r"docker.*not found|no such file.*docker|FileNotFoundError.*docker", re.IGNORECASE), "docker_not_found"),
    (re.compile(r"docker.*error|error.*docker", re.IGNORECASE), "docker_error"),
    (re.compile(r"evaluation\s*error", re.IGNORECASE), "evaluation_error"),
    (re.compile(r"out of memory|OOM|memory.*killed", re.IGNORECASE), "oom_error"),
    # Be specific about timeout errors - exclude normal "timeout -s" commands used in evaluators
    (re.compile(r"ERROR:.*timed?\s*out|execution timed out|process timed out", re.IGNORECASE), "timeout_error"),
    (re.compile(r"permission denied", re.IGNORECASE), "permission_denied"),
    (re.compile(r"connection refused|network.*error", re.IGNORECASE), "network_error"),
    (re.compile(r"ImportError|ModuleNotFoundError", re.IGNORECASE), "import_error"),
    (re.compile(r"CUDA.*error|GPU.*error|cuDNN.*error", re.IGNORECASE), "gpu_error"),
    # Syntax errors in generated solution code
    (re.compile(r"SyntaxError:|IndentationError:", re.IGNORECASE), "syntax_error"),
    # HTTP errors for missing data/resources
    (re.compile(r"HTTPError.*404|404.*not found", re.IGNORECASE), "http_404"),
    # Fatal evaluator errors
    (re.compile(r"\[Evaluator\] Fatal error:", re.IGNORECASE), "evaluator_fatal"),
]


def detect_content_warnings(text: str) -> List[str]:
    """Scan result file content for warning patterns."""
    warnings = []
    for pattern, warning_type in WARNING_PATTERNS:
        if pattern.search(text):
            warnings.append(warning_type)
    return warnings


@dataclass
class ResultSyncSummary:
    records: List[ResultRecord]
    missing_pairs: List[Pair]
    unexpected_files: List[Path]
    baseline_files: List[Path]  # Baseline files (may include deleted problems)
    duplicates: List[str]
    pair_coverage_warnings: List[str]
    total_expected_pairs: int


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


def detect_category_structure(problem_path: Path) -> Tuple[bool, List[str]]:
    """
    Detect if a problem directory is a category containing multiple variants.

    A category is a directory where:
    - It does NOT have its own evaluator.py at the top level
    - It has subdirectories that each contain evaluator.py

    Returns:
        (is_category, list_of_variant_subdirs)
    """
    if not problem_path.is_dir():
        return False, []

    # Check if this directory has its own evaluator.py
    if (problem_path / "evaluator.py").exists():
        return False, []

    # Check for variant subdirectories
    variants = []
    for subdir in sorted(problem_path.iterdir()):
        if subdir.is_dir() and (subdir / "evaluator.py").exists():
            variants.append(subdir.name)

    # It's a category if it has multiple variants
    if len(variants) > 1:
        return True, variants

    return False, []


def read_problem_list(path: Path) -> Tuple[List[str], dict]:
    """
    Read problems from file and expand categories into variants.

    Returns:
        (problems, category_map)
        - problems: List of all problem paths (including expanded variants)
        - category_map: Dict mapping variant paths to their category path
                       e.g., {'poc_gen/heap_overflow/arvo_47101': 'poc_gen/heap_overflow'}
    """
    problems = []
    category_map = {}  # variant_path -> category_path

    problems_root = path.parent / "problems"

    for entry in read_list_file(path):
        normalized = entry.split("research/", 1)[-1]
        problem_full_path = problems_root / normalized

        is_category, variants = detect_category_structure(problem_full_path)

        if is_category:
            # Expand category into individual variants
            for variant in variants:
                variant_path = f"{normalized}/{variant}"
                problems.append(variant_path)
                category_map[variant_path] = normalized
        else:
            # Regular problem, add as-is
            problems.append(normalized)

    return problems, category_map


def read_models_list(path: Path) -> List[str]:
    models: List[str] = []
    seen: set[str] = set()
    for entry in read_list_file(path):
        if entry not in seen:
            models.append(entry)
            seen.add(entry)
    return models


def read_variant_indices(path: Path) -> List[int]:
    """Read variant indices from file.

    - One integer per line; lines may be blank or commented with '#'.
    - 0 means no suffix; other numbers are used as `_N` suffix.
    - Backward compatible: if a single integer N is present, expand to [0..N-1].
    """
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

    # Distinguish GPT-5 minor versions so that 'gpt-5.1' does not
    # collapse into the same prefix as 'gpt-5'. Keep backward
    # compatibility for plain 'gpt-5' which maps to 'gpt5'.
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


def build_expected_pairs(
    problems: List[str],
    model_prefixes: List[str],
    variant_indices: List[int],
) -> List[Pair]:
    pairs: List[Pair] = []
    for problem in problems:
        slug = sanitize_problem_name(problem)
        for prefix in model_prefixes:
            base_solution = f"{prefix}_{slug}"
            for idx in variant_indices or [0]:
                suffix = "" if idx == 0 else f"_{idx}"
                solution = f"{base_solution}{suffix}"
                pairs.append(Pair(solution=solution, problem=problem))
    return pairs


def sanitize_problem_name(problem: str) -> str:
    return problem.replace("/", "_")


def expected_result_filename(solution: str, problem: str) -> str:
    return f"{solution}_{sanitize_problem_name(problem)}{RESULT_SUFFIX}"


def extract_solution_and_problem(
    lines: Iterable[str],
    *,
    default_solution: Optional[str],
    default_problem: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    solution = default_solution
    problem = default_problem

    for line in lines:
        if solution is None:
            match = SOLUTION_LINE_RE.search(line)
            if match:
                solution = match.group(1).strip().strip("'\"")
        if problem is None:
            match = PROBLEM_LINE_RE.search(line)
            if match:
                problem = match.group(1).strip().strip("'\"")
        if solution is not None and problem is not None:
            break

    return solution, problem


def extract_score_and_status(lines: Iterable[str]) -> Tuple[str, str]:
    cleaned = [line.strip() for line in lines if line.strip()]

    for line in reversed(cleaned):
        if "[INFO]" in line or "[DEBUG]" in line:
            continue
        if NUMERIC_LINE_RE.match(line):
            return line, "SUCCESS"

    for line in reversed(cleaned):
        if line.startswith("SKIP:"):
            return "N/A", line

    for line in reversed(cleaned):
        if line.startswith("ERROR:"):
            return "N/A", line
        if "Error:" in line or "Exception" in line or "failed" in line.lower():
            return "N/A", f"ERROR: {line}"

    for line in reversed(cleaned):
        if line.startswith("Traceback"):
            return "N/A", "ERROR: Traceback detected"

    if cleaned:
        snippet = cleaned[-1]
        if len(snippet) > 80:
            snippet = snippet[:77] + "..."
        return "N/A", f"ERROR: No score found (last line: {snippet})"

    return "N/A", "ERROR: Empty log"


def score_to_float(score: str) -> Optional[float]:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return value


def clamp_score(value: Optional[float]) -> tuple[Optional[float], Optional[float], bool]:
    """Clamp numeric score to [0, 100].

    Returns (clamped_value, original_value, was_clamped).
    If value is None, returns (None, None, False).
    """
    if value is None:
        return None, None, False
    orig = value
    clamped = min(100.0, max(0.0, value))
    return clamped, orig, (not math.isclose(clamped, orig))


def parse_result_file(
    path: Path,
    *,
    default_solution: Optional[str],
    default_problem: Optional[str],
) -> ResultRecord:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    solution, problem = extract_solution_and_problem(
        lines, default_solution=default_solution, default_problem=default_problem
    )

    score, status = extract_score_and_status(lines)
    timestamp = dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    numeric = score_to_float(score)
    clamped, original, was_clamped = clamp_score(numeric)

    # Detect potential issues in content even if score looks valid
    content_warnings = detect_content_warnings(text)

    return ResultRecord(
        solution=solution or (default_solution or "UNKNOWN"),
        problem=problem or (default_problem or "UNKNOWN"),
        score=score,
        status=status,
        timestamp=timestamp,
        filename=path.name,
        score_value=clamped,
        original_score_value=original,
        was_clamped=was_clamped,
        content_warnings=content_warnings,
    )


def _write_model_aggregates_csv(
    *,
    out_path: Path,
    records: Sequence[ResultRecord],
    problems: Sequence[str],
    model_prefixes: Sequence[str],
    variant_indices: List[int],
) -> None:
    """Compute per-model aggregates over all configured problems and save CSV.

    Metrics per model:
      - Score@1: average score using attempt #1 (index 0) per problem.
      - Avg@5: mean of up to first 5 attempts per problem, averaged over problems.
      - Score@5: best (max) of up to first 5 attempts per problem, averaged.
      - Pass@1: fraction of problems with non-zero score at attempt #1.
      - Pass@5: fraction of problems with any non-zero score among first 5 attempts.

    Notes:
      - Non-numeric / failed evaluations are treated as 0 for scoring and pass.
      - "Pass" follows the user definition: pass if score != 0.
    """

    denom5 = max(1, min(5, len(variant_indices)))

    # Pre-compute problem slugs for exact matching against solution field.
    problem_slugs = [sanitize_problem_name(p) for p in problems]
    slug_set = set(problem_slugs)

    # Map (model, problem_slug, position) -> numeric score (0 if N/A)
    # position is the ordinal in variant_indices (0-based), not the numeric suffix
    values: dict[Tuple[str, str, int], float] = {}
    pos_by_index = {idx: pos for pos, idx in enumerate(variant_indices)}

    for r in records:
        # Only consider configured models and problems
        model = r.solution.split("_", 1)[0]
        if model not in model_prefixes:
            continue
        prob_slug = sanitize_problem_name(r.problem)
        if prob_slug not in slug_set:
            continue

        base = f"{model}_{prob_slug}"
        if r.solution == base:
            idx = 0
        elif r.solution.startswith(base + "_"):
            suffix = r.solution[len(base) + 1 :]
            if not suffix.isdigit():
                continue
            idx = int(suffix)
        else:
            # Different pairing; skip
            continue
        # Remap to position; skip indices not configured
        if idx not in pos_by_index:
            continue
        pos = pos_by_index[idx]

        val = r.score_value if (r.status == "SUCCESS" and r.score_value is not None) else 0.0
        key = (model, prob_slug, pos)
        # If duplicates exist, keep the max value conservatively.
        if key in values:
            values[key] = max(values[key], val)
        else:
            values[key] = val

    # Aggregate per model over all configured problems
    rows: list[tuple[str, float, float, float, float, float, int]] = []
    problem_count = len(problem_slugs)
    if problem_count == 0:
        # Still write an empty file with header for robustness
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "Score@1", "Avg@5", "Score@5", "Pass@1", "Pass@5", "problems"])
        return

    for model in model_prefixes:
        sum_s1 = 0.0
        sum_avg5 = 0.0
        sum_s5 = 0.0
        pass1 = 0
        pass5 = 0

        for prob_slug in problem_slugs:
            attempt_vals = [values.get((model, prob_slug, i), 0.0) for i in range(denom5)]
            v1 = attempt_vals[0] if attempt_vals else 0.0
            avg5 = sum(attempt_vals) / float(denom5)
            best5 = max(attempt_vals) if attempt_vals else 0.0

            sum_s1 += v1
            sum_avg5 += avg5
            sum_s5 += best5
            pass1 += 1 if v1 != 0 else 0
            pass5 += 1 if any(v != 0 for v in attempt_vals) else 0

        rows.append(
            (
                model,
                sum_s1 / problem_count,
                sum_avg5 / problem_count,
                sum_s5 / problem_count,
                pass1 / problem_count,
                pass5 / problem_count,
                problem_count,
            )
        )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "Score@1", "Avg@5", "Score@5", "Pass@1", "Pass@5", "problems"])
        for (model, s1, a5, s5, p1, p5, pc) in rows:
            writer.writerow([
                model,
                f"{s1:.6f}",
                f"{a5:.6f}",
                f"{s5:.6f}",
                f"{p1:.6f}",
                f"{p5:.6f}",
                pc,
            ])


def _write_problem_model_matrix_csv(
    *,
    out_path: Path,
    records: Sequence[ResultRecord],
    problems: Sequence[str],
    model_prefixes: Sequence[str],
    variant_indices: List[int],
    category_map: dict[str, str],
) -> None:
    """Write a matrix CSV with problems as rows and models as columns.

    Each cell contains Score@5 (best of up to 5 attempts) for that (problem, model) pair.
    For categories, the score is averaged across all variants.
    """
    denom5 = max(1, min(5, len(variant_indices)))
    problem_slugs = [sanitize_problem_name(p) for p in problems]
    slug_set = set(problem_slugs)

    # Map (model, problem_slug, position) -> numeric score
    values: dict[Tuple[str, str, int], float] = {}
    pos_by_index = {idx: pos for pos, idx in enumerate(variant_indices)}

    for r in records:
        model = r.solution.split("_", 1)[0]
        if model not in model_prefixes:
            continue
        prob_slug = sanitize_problem_name(r.problem)
        if prob_slug not in slug_set:
            continue

        base = f"{model}_{prob_slug}"
        if r.solution == base:
            idx = 0
        elif r.solution.startswith(base + "_"):
            suffix = r.solution[len(base) + 1:]
            if not suffix.isdigit():
                continue
            idx = int(suffix)
        else:
            continue
        if idx not in pos_by_index:
            continue
        pos = pos_by_index[idx]

        val = r.score_value if (r.status == "SUCCESS" and r.score_value is not None) else 0.0
        key = (model, prob_slug, pos)
        if key in values:
            values[key] = max(values[key], val)
        else:
            values[key] = val

    def score5(model: str, slug: str) -> float:
        return max((values.get((model, slug, i), 0.0) for i in range(denom5)), default=0.0)

    # Aggregate by category if category_map is provided
    if category_map:
        # Get unique categories and non-category problems
        categories = sorted(set(category_map.values()))
        non_category_problems = [p for p in problems if p not in category_map]
        aggregated_problems = categories + non_category_problems

        # variant_problems[category] = list of variant problem paths
        variant_problems: dict[str, List[str]] = {}
        for variant, cat in category_map.items():
            variant_problems.setdefault(cat, []).append(variant)

        header = ["problem"] + list(model_prefixes)
        rows: list[list[str]] = []

        for prob in aggregated_problems:
            row = [prob]
            if prob in variant_problems:
                # Category: average across variants
                variants = variant_problems[prob]
                for model in model_prefixes:
                    variant_scores = [score5(model, sanitize_problem_name(v)) for v in variants]
                    avg_score = sum(variant_scores) / len(variant_scores) if variant_scores else 0.0
                    row.append(f"{avg_score:.6f}")
            else:
                # Non-category problem
                for model in model_prefixes:
                    s5 = score5(model, sanitize_problem_name(prob))
                    row.append(f"{s5:.6f}")
            rows.append(row)
    else:
        # No category aggregation
        header = ["problem"] + list(model_prefixes)
        rows = []
        for prob, slug in zip(problems, problem_slugs):
            row = [prob]
            for model in model_prefixes:
                s5 = score5(model, slug)
                row.append(f"{s5:.6f}")
            rows.append(row)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _write_problem_leaderboard_csv(
    *,
    out_problems_path: Path,
    out_summary_path: Path,
    records: Sequence[ResultRecord],
    problems: Sequence[str],
    model_prefixes: Sequence[str],
    variant_indices: List[int],
    focus_models: Sequence[str],
) -> None:
    # Build attempt values like in model aggregates
    denom5 = max(1, min(5, len(variant_indices)))
    problem_slugs = [sanitize_problem_name(p) for p in problems]
    slug_set = set(problem_slugs)

    values: dict[Tuple[str, str, int], float] = {}
    pos_by_index = {idx: pos for pos, idx in enumerate(variant_indices)}
    for r in records:
        model = r.solution.split("_", 1)[0]
        if model not in model_prefixes:
            continue
        prob_slug = sanitize_problem_name(r.problem)
        if prob_slug not in slug_set:
            continue

        base = f"{model}_{prob_slug}"
        if r.solution == base:
            idx = 0
        elif r.solution.startswith(base + "_"):
            suffix = r.solution[len(base) + 1 :]
            if not suffix.isdigit():
                continue
            idx = int(suffix)
        else:
            continue
        if idx not in pos_by_index:
            continue
        pos = pos_by_index[idx]

        val = r.score_value if (r.status == "SUCCESS" and r.score_value is not None) else 0.0
        key = (model, prob_slug, pos)
        if key in values:
            values[key] = max(values[key], val)
        else:
            values[key] = val

    # Per-problem rows
    header = [
        "problem",
        "top_naive_model",
        "top_naive_score",
        "top_overlap_model",
        "top_overlap_score",
    ]
    for m in focus_models:
        header += [f"Score@5_{m}", f"coverage_{m}"]

    rows: list[list[str]] = []

    def score5(model: str, slug: str) -> float:
        return max((values.get((model, slug, i), 0.0) for i in range(denom5)), default=0.0)

    def covered(model: str, slug: str) -> bool:
        return any(((model, slug, i) in values) for i in range(denom5))

    for slug in problem_slugs:
        # compute per-focus model metrics
        s5 = {m: score5(m, slug) for m in focus_models}
        cov = {m: covered(m, slug) for m in focus_models}

        # naive top: treat missing as 0
        top_naive_model = max(focus_models, key=lambda m: (s5[m], m)) if focus_models else ""
        top_naive_score = s5[top_naive_model] if top_naive_model else 0.0

        # overlap top: restrict to models with coverage on this problem; need >=2 to be meaningful
        overlap_models = [m for m in focus_models if cov[m]]
        if len(overlap_models) >= 2:
            top_overlap_model = max(overlap_models, key=lambda m: (s5[m], m))
            top_overlap_score = s5[top_overlap_model]
        else:
            top_overlap_model = ""
            top_overlap_score = 0.0

        row: list[str] = [slug, top_naive_model, f"{top_naive_score:.6f}", top_overlap_model, f"{top_overlap_score:.6f}"]
        for m in focus_models:
            row += [f"{s5[m]:.6f}", "1" if cov[m] else "0"]
        rows.append(row)

    with out_problems_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    # Leaderboard summary per model
    # - coverage_rate, top_naive_count, top_overlap_count, avg_score5_overlap
    # - pairwise wins/losses/ties on overlap-only comparisons
    totals = {
        m: {
            "cover": 0,
            "top_naive": 0,
            "top_overlap": 0,
            "sum_s5_overlap": 0.0,
            "cnt_s5_overlap": 0,
            "pair": {},  # opponent -> dict(win, loss, tie)
        }
        for m in focus_models
    }
    for m in focus_models:
        for n in focus_models:
            if m == n:
                continue
            totals[m]["pair"][n] = {"win": 0, "loss": 0, "tie": 0}

    for slug in problem_slugs:
        s5 = {m: score5(m, slug) for m in focus_models}
        cov = {m: covered(m, slug) for m in focus_models}
        for m in focus_models:
            totals[m]["cover"] += 1 if cov[m] else 0
            if cov[m]:
                totals[m]["sum_s5_overlap"] += s5[m]
                totals[m]["cnt_s5_overlap"] += 1
        top_naive = max(focus_models, key=lambda m: (s5[m], m)) if focus_models else None
        if top_naive is not None:
            totals[top_naive]["top_naive"] += 1
        overlap_models = [m for m in focus_models if cov[m]]
        if len(overlap_models) >= 2:
            top_overlap = max(overlap_models, key=lambda m: (s5[m], m))
            totals[top_overlap]["top_overlap"] += 1

    # Compute pairwise results
    for slug in problem_slugs:
        # compute once for this slug
        s5 = {m: score5(m, slug) for m in focus_models}
        cov = {m: covered(m, slug) for m in focus_models}
        for i, m in enumerate(focus_models):
            for n in focus_models[i + 1 :]:
                if cov[m] and cov[n]:
                    if s5[m] > s5[n]:
                        totals[m]["pair"][n]["win"] += 1
                        totals[n]["pair"][m]["loss"] += 1
                    elif s5[m] < s5[n]:
                        totals[m]["pair"][n]["loss"] += 1
                        totals[n]["pair"][m]["win"] += 1
                    else:
                        totals[m]["pair"][n]["tie"] += 1
                        totals[n]["pair"][m]["tie"] += 1

    with out_summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # dynamic header with pairwise columns
        header = [
            "model",
            "coverage_rate",
            "top_naive",
            "top_overlap",
            "avg_score5_overlap",
            "problems",
        ]
        for opp in focus_models:
            header += [f"wins_vs_{opp}", f"losses_vs_{opp}", f"ties_vs_{opp}"]
        writer.writerow(header)
        total_problems = len(problem_slugs)
        for m in focus_models:
            cover_rate = (totals[m]["cover"] / total_problems) if total_problems else 0.0
            avg_s5_overlap = (totals[m]["sum_s5_overlap"] / totals[m]["cnt_s5_overlap"]) if totals[m]["cnt_s5_overlap"] else 0.0
            row = [
                m,
                f"{cover_rate:.6f}",
                str(totals[m]["top_naive"]),
                str(totals[m]["top_overlap"]),
                f"{avg_s5_overlap:.6f}",
                str(total_problems),
            ]
            for opp in focus_models:
                if opp == m:
                    row += ["-", "-", "-"]
                else:
                    pair = totals[m]["pair"][opp]
                    row += [str(pair["win"]), str(pair["loss"]), str(pair["tie"])]
            writer.writerow(row)



def validate_expected_pairs(pairs: List[Pair]) -> List[str]:
    warnings: List[str] = []
    seen: set[Tuple[str, str]] = set()
    duplicates: List[str] = []
    for pair in pairs:
        key = (pair.solution, pair.problem)
        if key in seen:
            duplicates.append(f"{pair.solution}:{pair.problem}")
        else:
            seen.add(key)
    if duplicates:
        warnings.append(
            "Duplicate solution/problem entries detected: " + ", ".join(sorted(duplicates))
        )
    return warnings


def refresh_results_summary(
    results_dir: Path,
    problems_path: Path,
    models_path: Path,
    num_solutions_path: Path,
    *,
    csv_path: Path | None = None,
    models_csv_path: Path | None = None,
    backup: bool = True,
) -> ResultSyncSummary:
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_path or (results_dir / "results.csv")
    models_csv_path = models_csv_path or (results_dir / "models_summary.csv")

    category_map = {}  # variant_path -> category_path
    try:
        problems, category_map = read_problem_list(problems_path)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(f"Failed to process {problems_path}: {exc}")
        problems = []

    try:
        raw_models = read_models_list(models_path)
        model_prefixes = [canonical_model_prefix(model) for model in raw_models]
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(f"Failed to process {models_path}: {exc}")
        model_prefixes = []

    try:
        variant_indices = read_variant_indices(num_solutions_path)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(f"Failed to process {num_solutions_path}: {exc}")
        variant_indices = [0]

    # Emit a concise config summary
    logger.info(
        "CONFIG | results_dir=%s | problems=%s (count=%d) | models=%s (count=%d) | num_solutions=%s (indices=%s)",
        str(results_dir),
        str(problems_path),
        len(problems),
        str(models_path),
        len(model_prefixes),
        str(num_solutions_path),
        ",".join(str(i) for i in variant_indices),
    )

    coverage_warnings: List[str] = []
    pairs = (
        build_expected_pairs(problems, model_prefixes, variant_indices)
        if (problems and model_prefixes)
        else []
    )
    coverage_warnings.extend(validate_expected_pairs(pairs))

    sanitized_lookup: dict[str, str] = {}
    duplicates: List[str] = []
    for pair in pairs:
        sanitized_lookup.setdefault(sanitize_problem_name(pair.problem), pair.problem)

    for warning in coverage_warnings:
        logger.warning(f"Pair coverage: {warning}")

    raw_actual_files = {
        path.name: path for path in results_dir.glob(f"*{RESULT_SUFFIX}")
    }
    actual_files: dict[str, Path] = {}
    for name, path in raw_actual_files.items():
        remapped = False
        for old_prefix, new_prefix in MODEL_PREFIX_ALIASES.items():
            if name.startswith(old_prefix):
                alias_name = new_prefix + name[len(old_prefix):]
                actual_files[alias_name] = path
                remapped = True
                break
        if not remapped:
            actual_files[name] = path

    records: List[ResultRecord] = []
    matched_names: set[str] = set()
    missing_pairs: List[Pair] = []
    for pair in pairs:
        candidate_name = expected_result_filename(pair.solution, pair.problem)
        chosen_path = actual_files.get(candidate_name)
        if chosen_path is None:
            missing_pairs.append(pair)
            logger.error(
                f"Expected result missing: "
                f"{pair.solution}:{pair.problem} (looked for {candidate_name})"
            )
            continue
        matched_names.add(candidate_name)
        records.append(
            parse_result_file(
                chosen_path,
                default_solution=pair.solution,
                default_problem=pair.problem,
            )
        )

    baseline_files: List[Path] = []
    unexpected_files: List[Path] = []
    for name in sorted(actual_files):
        if name in matched_names:
            continue
        path = actual_files[name]
        if "_baseline" in path.stem:
            baseline_files.append(path)
        else:
            unexpected_files.append(path)
    for path in unexpected_files:
        logger.warning(f"Result file has no configured pair: {path.name}")

    leftovers = unexpected_files + baseline_files
    for path in leftovers:
        base = path.stem
        default_solution: Optional[str] = None
        default_problem: Optional[str] = None
        for sanitized, original in sorted(
            sanitized_lookup.items(), key=lambda item: len(item[0]), reverse=True
        ):
            suffix = f"_{sanitized}"
            if base.endswith(suffix):
                default_solution = base[: -len(suffix)] or None
                default_problem = original
                break
        records.append(
            parse_result_file(
                path,
                default_solution=default_solution,
                default_problem=default_problem,
            )
        )

    if csv_path.exists() and backup:
        backup_path = csv_path.with_suffix(csv_path.suffix + ".bak")
        shutil.copy2(csv_path, backup_path)

    def _fmt_score_for_csv(score_value: Optional[float], fallback: str = "N/A") -> str:
        if score_value is not None:
            if float(score_value).is_integer():
                return str(int(score_value))
            return f"{score_value:.6f}"
        return fallback

    def _aggregate_by_category(
        records: List[ResultRecord],
        category_map: dict,
    ) -> List[ResultRecord]:
        """
        Aggregate variant-level records into category-level records.

        For each (solution_base, category), compute the average score across all variants.
        Non-category records are passed through unchanged.
        """
        if not category_map:
            return records

        # Group records by (solution_base, category)
        # solution_base is the solution name without the problem suffix
        from collections import defaultdict

        # Records that don't belong to any category
        non_category_records = []
        # Records grouped by category: (solution_prefix, category) -> list of records
        category_groups: dict[Tuple[str, str], List[ResultRecord]] = defaultdict(list)

        for r in records:
            if r.problem in category_map:
                category = category_map[r.problem]
                # Extract solution prefix (model + variant index)
                # e.g., "gpt5_poc_generation_heap_buffer_overflow_arvo_47101_1"
                # -> we need "gpt5_1" as the solution base
                problem_slug = sanitize_problem_name(r.problem)
                if r.solution.endswith(f"_{problem_slug}"):
                    sol_prefix = r.solution[: -len(problem_slug) - 1]
                elif f"_{problem_slug}_" in r.solution:
                    # Handle case with variant index suffix
                    idx = r.solution.find(f"_{problem_slug}_")
                    sol_prefix = r.solution[:idx] + r.solution[idx + len(problem_slug) + 1:]
                else:
                    sol_prefix = r.solution
                category_groups[(sol_prefix, category)].append(r)
            else:
                non_category_records.append(r)

        # Aggregate each category group
        aggregated_records = []
        for (sol_prefix, category), group_records in category_groups.items():
            # Calculate average score
            valid_scores = [
                r.score_value for r in group_records
                if r.status == "SUCCESS" and r.score_value is not None
            ]

            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
                status = "SUCCESS"
                score_str = _fmt_score_for_csv(avg_score)
            else:
                avg_score = None
                status = "N/A"
                score_str = "N/A"

            # Use the most recent timestamp
            latest_timestamp = max(r.timestamp for r in group_records)

            # Create aggregated solution name
            category_slug = sanitize_problem_name(category)
            agg_solution = f"{sol_prefix}_{category_slug}"

            aggregated_records.append(ResultRecord(
                solution=agg_solution,
                problem=category,
                score=score_str,
                status=status,
                timestamp=latest_timestamp,
                filename=f"[aggregated from {len(group_records)} variants]",
                score_value=avg_score,
                original_score_value=avg_score,
                was_clamped=False,
                content_warnings=[],
            ))

        return non_category_records + aggregated_records

    # Write detailed results (variant-level) to a separate file
    detailed_csv_path = csv_path.with_name("results_detailed.csv")
    with detailed_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["solution", "problem", "score", "status", "timestamp"])
        for record in records:
            score_str = _fmt_score_for_csv(record.score_value, record.score) if record.status == "SUCCESS" else record.score
            writer.writerow([record.solution, record.problem, score_str, record.status, record.timestamp])

    # Write aggregated results (category-level) to main CSV
    aggregated_records = _aggregate_by_category(records, category_map)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["solution", "problem", "score", "status", "timestamp"])
        for record in aggregated_records:
            writer.writerow([record.solution, record.problem, record.score, record.status, record.timestamp])

    # Error logs for scores outside [0, 100] (after clamping)
    oob_events = [
        r for r in records
        if (r.original_score_value is not None and r.was_clamped and r.status == "SUCCESS")
    ]
    if oob_events:
        log_path = results_dir / "out_of_range_scores.log"
        with log_path.open("w", encoding="utf-8") as lf:
            lf.write("# Scores outside [0,100] were clamped; original -> clamped\n")
            lf.write("# filename,solution,problem,original,clamped,timestamp\n")
            for r in oob_events:
                lf.write(
                    f"{r.filename},{r.solution},{r.problem},{r.original_score_value},{r.score_value},{r.timestamp}\n"
                )
        for r in oob_events:
            logger.error(
                f"Out-of-range score clamped in {r.filename}: "
                f"{r.original_score_value} -> {r.score_value} ({r.solution}:{r.problem})"
            )

    # Also write per-model aggregate metrics across all configured problems.
    # Use category-level problems for aggregation if categories exist
    try:
        if category_map:
            # Get unique categories and non-category problems
            categories = set(category_map.values())
            non_category_problems = [p for p in problems if p not in category_map]
            aggregated_problems = list(categories) + non_category_problems
            records_for_model_agg = aggregated_records
        else:
            aggregated_problems = problems
            records_for_model_agg = records

        _write_model_aggregates_csv(
            out_path=models_csv_path,
            records=records_for_model_agg,
            problems=aggregated_problems,
            model_prefixes=model_prefixes,
            variant_indices=variant_indices,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(f"Failed to write models summary CSV: {exc}")

    return ResultSyncSummary(
        records=records,
        missing_pairs=missing_pairs,
        unexpected_files=unexpected_files,
        baseline_files=baseline_files,
        duplicates=duplicates,
        pair_coverage_warnings=coverage_warnings,
        total_expected_pairs=len(pairs),
    )


def _pairs_without_solutions(missing_pairs: Sequence[Pair], solutions_root: Path) -> List[Pair]:
    """Return the subset of missing pairs whose solutions/…/resources/solution.py is absent."""
    result: List[Pair] = []
    for pair in missing_pairs:
        sol_file = solutions_root / pair.solution / "resources" / "solution.py"
        if not sol_file.is_file():
            result.append(pair)
    return result


def format_record_lines(records: Sequence[ResultRecord], limit: int = 15) -> List[str]:
    lines: List[str] = []
    for idx, record in enumerate(records):
        if idx >= limit:
            remaining = len(records) - limit
            lines.append(f"  ... ({remaining} more)")
            break
        lines.append(
            f"  - {record.solution} -> {record.problem} "
            f"(score={record.score}, status={record.status})"
        )
    return lines


def visualize_summary(summary: ResultSyncSummary) -> None:
    """Print summary statistics to stdout (not logs)."""
    total_expected = summary.total_expected_pairs
    success_records = [r for r in summary.records if r.status == "SUCCESS" and r.score_value is not None]
    zero_records = [r for r in success_records if r.score_value == 0]
    negative_records = [r for r in success_records if r.score_value is not None and r.score_value < 0]
    na_records = [r for r in summary.records if r.score_value is None or r.status != "SUCCESS"]

    print("=== Results Sync Summary ===")
    print(f"Expected pairs (configured): {total_expected}")
    print(f"Completed result files:     {len(summary.records)}")
    print(f"Missing results:            {len(summary.missing_pairs)}")
    print(f"Unexpected result files:    {len(summary.unexpected_files)}")
    print(f"Duplicate entries:          {len(summary.duplicates)}")
    print()

    print(f"Successful numeric scores:  {len(success_records)}")
    print(f"  - Zero scores:            {len(zero_records)}")
    print(f"  - Negative scores:        {len(negative_records)}")
    print(f"Non-numeric / error scores: {len(na_records)}")
    print()

    if zero_records:
        print("Zero-score runs:")
        for line in format_record_lines(zero_records):
            print(line)
        print()

    if negative_records:
        print("Negative-score runs:")
        for line in format_record_lines(negative_records):
            print(line)
        print()

    if na_records:
        print("N/A or error runs:")
        for line in format_record_lines(na_records):
            print(line)
        print()

    if summary.missing_pairs:
        print("Missing result files for pairs:")
        for idx, pair in enumerate(summary.missing_pairs):
            if idx >= 15:
                remaining = len(summary.missing_pairs) - 15
                print(f"  ... ({remaining} more)")
                break
            print(f"  - {pair.solution}:{pair.problem}")
        print()

    if summary.unexpected_files:
        print("Result files without configured pair:")
        for idx, path in enumerate(summary.unexpected_files):
            if idx >= 15:
                remaining = len(summary.unexpected_files) - 15
                print(f"  ... ({remaining} more)")
                break
            print(f"  - {path.name}")
        print()

    model_totals: dict[str, List[float]] = {}
    for record in success_records:
        model_name = record.solution.split("_", 1)[0]
        model_totals.setdefault(model_name, []).append(record.score_value or 0.0)

    if model_totals:
        print("Per-model average scores (success cases):")
        print(f"{'Model':<15}{'Count':>7}{'Average':>12}")
        for model, values in sorted(
            model_totals.items(),
            key=lambda item: (-sum(item[1]) / max(len(item[1]), 1), item[0]),
        ):
            count = len(values)
            avg = sum(values) / count if count else 0.0
            print(f"{model:<15}{count:>7}{avg:>12.4f}")
        print()

    if summary.pair_coverage_warnings:
        print("Configuration warnings:")
        for warning in summary.pair_coverage_warnings:
            print(f"  - {warning}")
        print()

    # Show records with content warnings (potential evaluator issues)
    # Group by warning type for cleaner output
    # Skip unexpected files and baseline files (results from deleted problems)
    skip_filenames = {p.name for p in summary.unexpected_files} | {p.name for p in summary.baseline_files}
    warning_by_type: dict[str, List[ResultRecord]] = {}
    for r in summary.records:
        if r.filename in skip_filenames:
            continue  # Skip results from deleted/unknown problems
        if r.content_warnings:
            for w in r.content_warnings:
                warning_by_type.setdefault(w, []).append(r)

    if warning_by_type:
        print("=== Content Warnings (potential evaluator issues) ===")
        for warning_type, records_with_warning in sorted(warning_by_type.items()):
            # Separate into those with valid scores vs N/A
            valid_score = [r for r in records_with_warning if r.status == "SUCCESS"]
            na_score = [r for r in records_with_warning if r.status != "SUCCESS"]
            print(f"\n{warning_type}: {len(records_with_warning)} files")
            if valid_score:
                print(f"  With numeric score ({len(valid_score)}):")
                for idx, r in enumerate(valid_score[:10]):
                    print(f"    - {r.filename} (score={r.score})")
                if len(valid_score) > 10:
                    print(f"    ... ({len(valid_score) - 10} more)")
            if na_score:
                print(f"  With N/A score ({len(na_score)}):")
                for idx, r in enumerate(na_score[:5]):
                    print(f"    - {r.filename}")
                if len(na_score) > 5:
                    print(f"    ... ({len(na_score) - 5} more)")
        print()

    print("Sync complete. CSV refreshed with latest data.")


def _find_default_path(candidates: List[str]) -> str:
    """Find the first existing path from candidates."""
    for path in candidates:
        if Path(path).exists():
            return path
    return candidates[0]  # Return first candidate as default


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate results.csv and summarize scores.")
    parser.add_argument(
        "--results-dir",
        default=_find_default_path(["results", "../results"]),
        help="Directory containing *_result.txt logs (default: results or ../results)"
    )
    parser.add_argument(
        "--problems",
        default=_find_default_path(["research/problems.txt", "problems.txt"]),
        help="Path to problems.txt"
    )
    parser.add_argument(
        "--models",
        default=_find_default_path(["research/models.txt", "models.txt"]),
        help="Path to models.txt"
    )
    parser.add_argument(
        "--num-solutions",
        default=_find_default_path(["research/num_solutions.txt", "num_solutions.txt"]),
        help=(
            "Path to num_solutions.txt listing variant indices per line "
            "(0 means no suffix). Backward-compatible with a single integer N."
        ),
    )
    parser.add_argument("--csv-path", help="Override output CSV path (defaults to <results-dir>/results.csv)")
    parser.add_argument(
        "--models-csv-path",
        help="Output path for per-model aggregates CSV (defaults to <results-dir>/models_summary.csv)",
    )
    parser.add_argument(
        "--problems-csv-path",
        help="Output path for per-problem leaderboard CSV (defaults to <results-dir>/problem_leaderboard.csv)",
    )
    parser.add_argument(
        "--leaderboard-summary-csv",
        help="Output path for leaderboard summary CSV (defaults to <results-dir>/leaderboard_summary.csv)",
    )
    parser.add_argument(
        "--focus-models",
        default="gemini2.5pro,gpt5,grok4fastreasoning",
        help="Comma-separated model prefixes to focus on for leaderboard (default: gemini2.5pro,gpt5,grok4fastreasoning)",
    )
    parser.add_argument("--no-backup", action="store_true", help="Skip creating a .bak backup of the previous CSV")
    parser.add_argument("--no-visualize", action="store_true", help="Do not print the textual visualization summary")
    parser.add_argument(
        "--write-missing-pairs",
        nargs="?",
        const="skypilot_pending_pairs.txt",
        default=None,
        metavar="PATH",
        help=(
            "Write missing pairs to PATH; if PATH omitted, defaults to skypilot_pending_pairs.txt"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    results_dir = Path(args.results_dir)
    problems_path = Path(args.problems)
    models_path = Path(args.models)
    num_solutions_path = Path(args.num_solutions)
    csv_path = Path(args.csv_path) if args.csv_path else None
    models_csv_path = Path(args.models_csv_path) if args.models_csv_path else None
    problems_csv_path = Path(args.problems_csv_path) if args.problems_csv_path else None
    leaderboard_summary_csv = Path(args.leaderboard_summary_csv) if args.leaderboard_summary_csv else None
    focus_models = [m.strip() for m in (args.focus_models or '').split(',') if m.strip()]

    summary = refresh_results_summary(
        results_dir,
        problems_path,
        models_path,
        num_solutions_path,
        csv_path=csv_path,
        models_csv_path=models_csv_path,
        backup=not args.no_backup,
    )

    # Always update 'skipped_failed_solutions.txt' with pairs that have no local solutions.
    try:
        repo_root = problems_path.parent
        solutions_root = repo_root / "solutions"
        nosol_pairs = _pairs_without_solutions(summary.missing_pairs, solutions_root)
        skipped_path = repo_root / "skipped_failed_solutions.txt"
        lines = sorted({f"{p.solution}:{p.problem}" for p in nosol_pairs})
        skipped_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        logger.info(
            f"Updated {skipped_path.name}: {len(lines)} no-solution pairs."
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Failed to update skipped_failed_solutions.txt: {exc}")

    if args.write_missing_pairs is not None:
        pending_path = Path(args.write_missing_pairs)
        lines = [f"{p.solution}:{p.problem}" for p in summary.missing_pairs]
        lines = sorted(set(lines))
        pending_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        logger.info(f"Wrote {len(lines)} missing pairs to {pending_path}.")

    # Write per-problem leaderboard and summary
    try:
        problems, category_map = read_problem_list(problems_path)
        raw_models = read_models_list(models_path)
        model_prefixes = [canonical_model_prefix(m) for m in raw_models]
        variant_indices = read_variant_indices(num_solutions_path)
        out_problems = problems_csv_path or (results_dir / "problem_leaderboard.csv")
        out_summary = leaderboard_summary_csv or (results_dir / "leaderboard_summary.csv")
        if not focus_models:
            focus_models = ["gemini2.5pro", "gpt5", "grok4fastreasoning"]
        _write_problem_leaderboard_csv(
            out_problems_path=out_problems,
            out_summary_path=out_summary,
            records=summary.records,
            problems=problems,
            model_prefixes=model_prefixes,
            variant_indices=variant_indices,
            focus_models=focus_models,
        )
        logger.info(f"Wrote per-problem leaderboard to {out_problems} and summary to {out_summary}.")

        # Write problem x model matrix CSV
        matrix_csv_path = results_dir / "problem_model_matrix.csv"
        _write_problem_model_matrix_csv(
            out_path=matrix_csv_path,
            records=summary.records,
            problems=problems,
            model_prefixes=model_prefixes,
            variant_indices=variant_indices,
            category_map=category_map,
        )
        logger.info(f"Wrote problem x model matrix to {matrix_csv_path}.")
    except Exception as exc:
        logger.warning(f"Failed to write per-problem leaderboard: {exc}")

    if not args.no_visualize:
        visualize_summary(summary)

    # Generate summary.md from config.yaml tags
    try:
        summary_md_path = problems_path.parent / "summary.md"
        generate_summary_md(problems_path, summary_md_path)
    except Exception as exc:
        logger.warning(f"Failed to generate summary.md: {exc}")

    return 0


def _load_config_tag(problem_path: Path) -> Optional[str]:
    """Load the 'tag' field from a problem's config.yaml."""
    config_path = problem_path / "config.yaml"
    if not config_path.exists():
        return None

    try:
        content = config_path.read_text(encoding="utf-8")
        # Try JSON first
        if content.strip().startswith("{"):
            import json
            config = json.loads(content)
            return config.get("tag")
        else:
            # YAML format - simple parsing
            for line in content.splitlines():
                if line.startswith("tag:"):
                    return line.split(":", 1)[1].strip().strip('"\'')
    except Exception:
        pass
    return None


def generate_summary_md(problems_path: Path, output_path: Path) -> None:
    """Generate summary.md from config.yaml tags."""
    TAG_LABELS = {
        "os": "Operating Systems",
        "hpc": "HPC / GPU Systems",
        "ai": "Artificial Intelligence",
        "db": "Databases",
        "pl": "Programming Languages",
        "security": "Security",
    }
    TAG_ORDER = ["os", "hpc", "ai", "db", "pl", "security"]

    problems_root = problems_path.parent

    # Read problems and get their tags
    problems_by_tag: dict[str, List[str]] = {tag: [] for tag in TAG_ORDER}

    try:
        problems, _ = read_problem_list(problems_path)
    except Exception as exc:
        logger.warning(f"Failed to read problems list: {exc}")
        return

    for problem in problems:
        problem_full_path = problems_root / problem
        tag = _load_config_tag(problem_full_path)
        if tag and tag in problems_by_tag:
            problems_by_tag[tag].append(problem)
        else:
            logger.warning(f"Problem {problem} has unknown or missing tag: {tag}")

    # Generate markdown
    lines = ["# Problems", ""]
    for tag in TAG_ORDER:
        probs = sorted(problems_by_tag[tag])
        if probs:
            label = TAG_LABELS.get(tag, tag.upper())
            lines.append(f"- {label} ({len(probs)})")
            for p in probs:
                lines.append(f"    - {p}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Generated {output_path} with {sum(len(v) for v in problems_by_tag.values())} problems.")


if __name__ == "__main__":
    raise SystemExit(main())
