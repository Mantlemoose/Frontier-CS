# Frontier-CS

Benchmarks for evaluating LLMs on challenging computer science problems.

- **Research Problems** (49): Real-world systems challenges — GPU kernels, distributed scheduling, ML pipelines, security exploits
- **Algorithmic Problems** (107): Competitive programming challenges — optimization, construction, interactive

## Installation

```bash
git clone https://github.com/xxx/Frontier-CS.git
cd Frontier-CS

# Install Python dependencies (using uv, recommended)
uv sync

# Or with pip:
pip install -e .
```

### API Keys (for LLM evaluation)

Set environment variables for the models you want to use:

```bash
export OPENAI_API_KEY="sk-..."        # For GPT models
export ANTHROPIC_API_KEY="sk-ant-..." # For Claude models
export GOOGLE_API_KEY="..."           # For Gemini models
```

---

## Repository Structure

```
Frontier-CS/
├── research/           # Research problems (49 problems)
│   ├── flash_attn/
│   ├── gemm_optimization/
│   ├── cant_be_late/
│   └── ...
└── algorithmic/        # Algorithmic problems (107 problems)
    ├── problems/
    └── scripts/
```

---

## Research Problems

Real-world systems problems requiring domain expertise.

### Categories

| Category | Count | Examples |
|----------|-------|----------|
| OS / Distributed | 8 | `cant_be_late`, `cant_be_late_multi` |
| HPC / GPU | 19 | `flash_attn`, `gemm_optimization`, `cross_entropy` |
| AI / ML | 6 | `imagenet_pareto`, `cloudcast` |
| Database | 7 | `vdb_pareto`, `llm_sql` |
| PL | 5 | `symbolic_regression` |
| Security | 4 | `poc_generation` |

### Quick Start

```bash
cd research

# 1. Generate solutions with an LLM
python generate_oneshot_gpt.py --model gpt-5

# 2. Run evaluation locally (requires Docker)
./main_loop.sh

# Or run on cloud (requires SkyPilot)
python scripts/skypilot_per_solution.py --max-concurrent 4
```

See [research/CONTRIBUTING.md](research/CONTRIBUTING.md) for detailed usage.

---

## Algorithmic Problems

107 competitive programming problems with automated judging.

### Categories

| Category | Count | Description |
|----------|-------|-------------|
| Optimization | 29 | Find optimal solution for given constraints |
| Construction | 27 | Build a solution satisfying all constraints |
| Interactive | 51 | Query-response with judge |

### Quick Start

```bash
cd algorithmic

# 1. Start judge server (requires Docker)
docker-compose up -d

# 2. Run benchmark
python scripts/run_tests.py claude-opus-4-5
```

See [algorithmic/README.md](algorithmic/README.md) for details.

---

## Requirements

- **Python 3.12+**
- **Docker** (for local evaluation)
- **GPU** (optional, for GPU problems)
- **SkyPilot** (optional, for cloud evaluation)
- **API Keys** (for LLM solution generation)
