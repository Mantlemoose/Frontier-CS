#!/usr/bin/env python3
"""Parse generation logs to extract skipped and failed solutions."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def compute_problem_name(problem_path: str) -> str:
    path = Path(problem_path)
    parts = [p for p in path.parts if p != "problems"]
    if not parts:
        return ""
    if len(parts) >= 2:
        base = parts[0].replace("_pareto", "").replace("_multi", "")
        return f"{base}_{parts[1]}"
    return parts[0].replace("_pareto", "")

FAILED_PATTERN = re.compile(r"Failed \d+ solution\(s\):\s*(.*)")
SKIPPED_PATTERN = re.compile(r"Skipped \d+ existing solution\(s\):\s*(.*)")


def parse_list(block: str) -> List[str]:
    if not block:
        return []
    items = []
    for part in block.split(","):
        name = part.strip()
        if name:
            if " (" in name:
                name = name.split(" (", 1)[0].strip()
            if ". Use " in name:
                name = name.split(". Use ", 1)[0].strip()
            if name.endswith("."):
                name = name[:-1].strip()
            items.append(name)
    return items


def extract_sections(text: str) -> Tuple[List[str], List[str]]:
    skipped: List[str] = []
    failed: List[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Skipped "):
            match = SKIPPED_PATTERN.match(line)
            if match:
                skipped.extend(parse_list(match.group(1)))
        elif line.startswith("Failed "):
            match = FAILED_PATTERN.match(line)
            if match:
                failed.extend(parse_list(match.group(1)))
    return skipped, failed


def sanitize_problem_name(problem: str) -> str:
    return problem.replace("/", "_")


def build_problem_lookup(problems_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for raw_line in problems_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        normalized = normalize_problem_path(line)
        sanitized = sanitize_problem_name(normalized.split("research/", 1)[-1])
        mapping[sanitized] = normalized
    return mapping


def normalize_problem_path(problem: str) -> str:
    if problem.startswith("research/"):
        return problem
    return f"research/{problem.lstrip('./')}"


def infer_problem_from_solution(solution_name: str, lookup: Dict[str, str]) -> Optional[str]:
    base = solution_name
    if "_" in base:
        suffix_candidate = base.rsplit("_", 1)[1]
        if suffix_candidate.isdigit():
            base = base[: -len(suffix_candidate) - 1]

    for sanitized, problem in lookup.items():
        if base.endswith(sanitized):
            return problem
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract skipped and failed solutions from a generation log.")
    parser.add_argument("log_path", help="Path to gen_logs.md or similar log file")
    parser.add_argument("output_path", help="Path to write the combined solutions list")
    parser.add_argument("--problems", default="research/problems.txt", help="Path to problems.txt (default: problems.txt)")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.is_file():
        raise SystemExit(f"Log file not found: {log_path}")

    text = log_path.read_text(encoding="utf-8", errors="replace")
    skipped, failed = extract_sections(text)

    if not skipped and not failed:
        raise SystemExit("No skipped or failed solutions found in the log.")

    combined = []
    seen = set()
    for name in skipped + failed:
        if name in seen:
            continue
        seen.add(name)
        combined.append(name)

    output_lines = []
    missing = []

    problem_name_map: Dict[str, str] = {}
    problems_path = Path(args.problems)
    if problems_path.is_file():
        problem_lookup = build_problem_lookup(problems_path)
        for sanitized, problem_rel in problem_lookup.items():
            problem_name = compute_problem_name(problem_rel)
            if problem_name:
                problem_name_map.setdefault(problem_name, problem_rel)
    else:
        raise SystemExit(f"Problems file not found: {problems_path}")

    for solution_name in combined:
        problem_path = infer_problem_from_solution(solution_name, problem_lookup)

        if problem_path is None:
            base_solution = solution_name
            if "_" in base_solution:
                candidate_suffix = base_solution.split("_", 1)[-1]
                inferred = problem_name_map.get(candidate_suffix)
                if inferred:
                    problem_path = inferred

        if problem_path is None:
            missing.append(solution_name)
            output_lines.append(f"{solution_name}:UNKNOWN")
        else:
            output_lines.append(f"{solution_name}:{problem_path}")

    output_path = Path(args.output_path)
    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(output_lines)} entries to {output_path}.")
    if missing:
        print(
            "Warning: "
            f"{len(missing)} solution(s) could not be mapped to a problem: {', '.join(missing[:10])}"
        )


if __name__ == "__main__":
    main()
