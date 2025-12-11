# Research Problems: Contributing & Evaluation Guide

## Problem Structure

Each problem follows a standardized interface:

```
research/{problem}/
├── config.yaml          # Dependencies, datasets, runtime config
├── set_up_env.sh        # Environment setup
├── evaluate.sh          # Evaluation entry point
├── evaluator.py         # Scoring logic
├── readme               # Problem description
└── resources/           # Problem-specific code/data
```

### Solution Interface

Solutions implement a `Solution` class in `solution.py`:

```python
class Solution:
    def __init__(self):
        pass

    def solve(self, *args):
        # Returns: solution output (format varies by problem)
        pass
```

### Evaluation Flow

```
config.yaml → set_up_env.sh → solve.sh → evaluate.sh → evaluator.py → score (0-100)
```

---

## Adding a New Problem

### 1. Create Problem Directory

```bash
mkdir -p research/{problem_name}/resources
```

### 2. Create `config.yaml`

```yaml
tag: hpc                   # Category: os, hpc, ai, db, pl, security

dependencies:
  uv_project: resources    # Optional: uv project in resources/

datasets: []               # Optional: dataset URLs

runtime:
  timeout_seconds: 1800    # Evaluation timeout
  requires_gpu: true       # GPU requirement
  resources:               # SkyPilot resources
    accelerators: "L4:1"
    cpus: "8+"
    memory: "32+"
  environment: "CUDA 12.2, Python 3.11, PyTorch 2.0+"
```

### 3. Create Evaluation Scripts

**`set_up_env.sh`**: Prepare environment
```bash
#!/bin/bash
# Install dependencies, download data, etc.
```

**`evaluate.sh`**: Run evaluation
```bash
#!/bin/bash
python evaluator.py
```

**`evaluator.py`**: Score the solution (last line must be numeric score)
```python
# ... evaluation logic ...
print(score)  # Must be last line!
```

### 4. Register the Problem

Add to `research/problems.txt`:
```
research/{problem_name}
```

---

## Problem Variants

For problems with multiple configurations:

```
research/gemm_optimization/
├── squares/           # Variant 1
│   ├── config.yaml
│   └── evaluator.py
├── rectangles/        # Variant 2
└── transformerish/    # Variant 3
```

Register each variant in `problems.txt`:
```
research/gemm_optimization/squares
research/gemm_optimization/rectangles
```

---

## Running Evaluations

### Local Evaluation

```bash
cd research
./main_loop.sh
```

Reads `pairs.txt` for solution-problem pairs, runs in Docker containers.

### Cloud Evaluation (SkyPilot)

```bash
python scripts/skypilot_per_solution.py --max-concurrent 4
```

Options:
- `--max-concurrent N`: Parallel machines
- `--region`, `--cpus`, `--accelerators`: Override resources
- `--dry-run`: Prepare without launching

### Generating LLM Solutions

```bash
# Single problem
python generate_oneshot_gpt.py research/flash_attn --model gpt-5

# All problems
python generate_oneshot_gpt.py --model gpt-5

# Multiple variants
python generate_oneshot_gpt.py --model gpt-5 --variants 5
```

---

## Configuration Files

### `problems.txt`
List of problems to evaluate:
```
research/flash_attn
research/gemm_optimization/squares
research/cant_be_late/high_availability_loose_deadline
```

### `pairs.txt`
Solution-problem pairs:
```
gpt5_flash_attn:research/flash_attn
claude_gemm_squares:research/gemm_optimization/squares
```

### `docker_images.txt`
Custom Docker images per problem:
```
# Format: problem=image,gpu_flag,dind_flag
gemm_optimization=andylizf/triton-tlx:tlx-nv-cu122,gpu
poc_generation=python:3.11-slim-trixie,nogpu,dind
```

### `models.txt`
Models to generate solutions for:
```
gpt-5
claude-opus-4-5
gemini-2.5-pro
```

### `num_solutions.txt`
Variant indices to generate (one per line):
```
0
1
2
```

---

## Results

### Output Files
- `results/{solution}_{problem}_result.txt`: Individual results
- `results/results.csv`: Aggregated scores
- `results/summary.txt`: Summary statistics

### Syncing Results

```bash
python scripts/results_sync.py --results-dir results
```

Rebuilds CSV, detects missing results, computes averages per model.

---

## Workflow: Handling Failures

`results_sync.py` generates two files for re-running failed/missing evaluations:

### `skipped_failed_solutions.txt`
Solutions missing from `solutions/` directory. Regenerate them:
```bash
python generate_oneshot_gpt.py --solutions-file skipped_failed_solutions.txt
```

### `skypilot_pending_pairs.txt`
Solution-problem pairs without result files. Run evaluations:
```bash
python scripts/skypilot_per_solution.py --pairs-file skypilot_pending_pairs.txt --max-concurrent 4
```

### Typical Workflow

```bash
# 1. Generate solutions
python generate_oneshot_gpt.py --model gpt-5

# 2. Run evaluations
python scripts/skypilot_per_solution.py --max-concurrent 4

# 3. Sync results (generates skipped/pending files)
python scripts/results_sync.py --results-dir results

# 4. Regenerate failed solutions
python generate_oneshot_gpt.py --solutions-file skipped_failed_solutions.txt

# 5. Re-run pending evaluations
python scripts/skypilot_per_solution.py --pairs-file skypilot_pending_pairs.txt
```

---

## Result File Format

**Important**: The last line of each result file must be a numeric score.

```python
# evaluator.py
# ... evaluation logic ...
print(score)  # e.g., 85.5
```

If the last line is not numeric, the result is marked as error.

---

## Scripts Reference

All scripts are in `research/scripts/`.

### Core Scripts

| Script | Description |
|--------|-------------|
| `skypilot_per_solution.py` | Run evaluations on SkyPilot (cloud) |
| `results_sync.py` | Sync results, rebuild CSV, generate pending/failed lists |
| `check_solution_matrix.py` | Verify solutions/ directory coverage |

### Generation & Submission

| Script | Description |
|--------|-------------|
| `generate_oneshot_gpt.py` | Generate solutions using LLMs (in parent dir) |
| `submit.py` | Submit solution to evaluation server |
| `fetch.py` | Fetch evaluation result from server |
| `setup.py` | Upload problem to evaluation server |

### Utilities

| Script | Description |
|--------|-------------|
| `extract_failed_skipped.py` | Parse generation logs to extract failed/skipped solutions |
| `config_loader.py` | Internal library for loading config.yaml |

### Usage Examples

```bash
# Check solution coverage
python scripts/check_solution_matrix.py

# Extract failures from generation log
python scripts/extract_failed_skipped.py gen_logs.md

# Bulk submit solutions
python scripts/submit.py --submissions submissions/ --out sid_map.json

# Fetch results by sid map
python scripts/fetch.py --map sid_map.json --out results.json
```
