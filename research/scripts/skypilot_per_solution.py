#!/usr/bin/env python3
"""Launch per-solution SkyPilot jobs to evaluate Frontier-CS pairs in parallel.

This script prepares a per-solution workspace, builds a SkyPilot task via the
Python API, launches it on a dedicated cluster for each solution/problem pair,
and, once finished, pulls back the evaluation artifacts before tearing the
cluster down.

Usage (from repo root):
    python scripts/skypilot_per_solution.py --max-concurrent 4
"""

import argparse
import hashlib
import io
import logging
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import sky
from tqdm import tqdm

from config_loader import load_runtime_config, RuntimeConfig

try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False

BASE_DIR = Path(__file__).resolve().parents[1]
progress_bar: Optional[tqdm] = None


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

    @property
    def job_name(self) -> str:
        base = f"{self.solution}-{self.problem}"
        digest = hashlib.md5(base.encode("utf-8")).hexdigest()[:8]
        sanitized_base = sanitize_name(base)
        suffix = f"-{digest}"
        max_len = 63
        available = max_len - len(suffix)
        trimmed_base = sanitized_base[:available].rstrip("-")
        return sanitize_name(f"{trimmed_base}{suffix}")


@dataclass
class JobContext:
    pair: Pair
    job_dir: Path
    cluster_name: str
    task: sky.Task
    ordinal: int
    total: int
    request_id: Optional[str] = None
    launched: bool = False
    torn_down: bool = False
    handle: Optional[object] = None


def format_progress(ctx: JobContext, stage: str, message: str = "") -> None:
    """Format and log progress messages, respecting the progress bar if active."""
    prefix = f"[{ctx.ordinal:03}/{ctx.total:03}]"
    stage_block = stage.upper().ljust(7) if stage else ""
    text = f"{prefix} {stage_block} {message}".rstrip()
    if progress_bar is not None:
        progress_bar.write(text)
    else:
        logger.info(text)


def load_docker_gpu_config(config_path: Path) -> dict[str, tuple[Optional[str], bool, bool]]:
    """Load docker/GPU/dind configuration from config file.

    Returns:
        Dict mapping problem name to (image, gpu_required, dind_required) tuple.
    """
    config: dict[str, tuple[Optional[str], bool, bool]] = {}
    if not config_path.is_file():
        return config
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        image_spec = value.strip()
        if not key:
            continue
        image: Optional[str] = None
        gpu_required = False
        dind_required = False
        if image_spec:
            parts = [part.strip() for part in image_spec.split(",") if part.strip()]
            if parts:
                image = parts[0]
            if len(parts) > 1:
                flag = parts[1].lower()
                gpu_required = flag in {"gpu", "true", "1"}
            if len(parts) > 2:
                dind_flag = parts[2].lower()
                dind_required = dind_flag in {"dind", "docker", "true", "1"}
        config[key] = (image, gpu_required, dind_required)
    return config


def sanitize_name(name: str) -> str:
    cleaned = []
    valid = "abcdefghijklmnopqrstuvwxyz0123456789-"
    last_dash = False
    for ch in name.lower():
        if ch in valid:
            cleaned.append(ch)
            last_dash = ch == "-"
        else:
            if not last_dash:
                cleaned.append("-")
                last_dash = True
    sanitized = "".join(cleaned).strip("-")
    return sanitized or "job"


def get_problem_name(problem_path: Path) -> str:
    """
    Extract the problem name from the problem path.
    Examples:
    - research/vdb_pareto/balanced -> vdb_pareto_balanced
    - research/poc_generation/heap_buffer_overflow/arvo_47101 -> poc_generation_heap_buffer_overflow_arvo_47101
    """
    parts = [p for p in problem_path.parts if p and p != "problems"]
    if not parts:
        raise ValueError(f"Unable to derive problem name from '{problem_path}'")
    return "_".join(parts)


def get_model_prefix(model: str) -> str:
    """
    Convert model name to the prefix format used in solution folder names.

    This MUST match the logic in generate_oneshot_gpt.py to ensure solution
    folder names are consistent between generation and evaluation.

    Examples:
    - 'gpt-5' or 'gpt-5-*' -> 'gpt5'
    - 'gpt-5.1' -> 'gpt5.1'
    - 'gemini/gemini-2.5-pro' -> 'gemini2.5pro'
    - 'anthropic/claude-sonnet-4-5-20250929' -> 'claude4.5sonnet'
    """
    import re

    # Remove provider prefix if present (e.g., 'gemini/gemini-2.5-pro' -> 'gemini-2.5-pro')
    if "/" in model:
        model = model.split("/", 1)[1]

    model_lower = model.lower().strip()

    # Handle GPT-5 variants - order matters! More specific first.
    if model_lower.startswith("gpt-5.1") or model_lower.startswith("gpt5.1"):
        return "gpt5.1"
    if model_lower.startswith("gpt-5") or model_lower.startswith("gpt5"):
        return "gpt5"

    # Handle Gemini 2.5 Pro variants
    if "gemini-2.5-pro" in model_lower or "gemini2.5pro" in model_lower:
        return "gemini2.5pro"

    # Handle other Gemini variants (e.g., gemini-1.5-pro -> gemini1.5pro, gemini-3-pro -> gemini3pro)
    gemini_match = re.match(r"gemini-?(\d+\.?\d*)-?pro", model_lower)
    if gemini_match:
        version = gemini_match.group(1)
        return f"gemini{version}pro"

    # Handle Claude variants (e.g., claude-sonnet-4-5-20250929 -> claude4.5sonnet)
    claude_match = re.match(r"claude-([a-z]+)-(\d+)-(\d+)", model_lower)
    if claude_match:
        family = claude_match.group(1)
        major = claude_match.group(2)
        minor = claude_match.group(3)
        return f"claude{major}.{minor}{family}"

    # Default: sanitize by removing all non-alphanumeric characters
    # This matches generate_oneshot_gpt.py's fallback behavior
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "", model_lower)
    return sanitized or "model"


def read_models_file(path: Path) -> List[str]:
    """Read models from models.txt file."""
    models = []
    if not path.exists():
        return models
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            models.append(line)
    return models


def read_num_solutions_file(path: Path) -> List[int]:
    """Read variant indices from num_solutions.txt."""
    indices = []
    if not path.exists():
        return [0]  # Default: just index 0 (no suffix)
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            try:
                indices.append(int(line))
            except ValueError:
                pass
    return indices if indices else [0]


def read_problem_list(path: Path) -> List[str]:
    """Read problem paths from a problem list file."""
    problems = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            # Normalize: remove 'research/' prefix if present
            if line.startswith("research/"):
                line = line[len("research/"):]
            problems.append(line)
    return problems


def expand_pairs_from_problems(
    problem_list_path: Path,
    models_path: Path,
    num_solutions_path: Path,
    validate_paths: bool = True
) -> List[Pair]:
    """
    Expand a problem list into solution:problem pairs using models and variants.

    For each problem, generates pairs for all models × all variants.
    Solution name format: {model_prefix}_{problem_name}[_{variant}]
    """
    problems = read_problem_list(problem_list_path)
    models = read_models_file(models_path)
    variant_indices = read_num_solutions_file(num_solutions_path)

    if not problems:
        raise ValueError(f"No problems found in {problem_list_path}")
    if not models:
        raise ValueError(f"No models found in {models_path}")

    logger.info(f"Expanding {len(problems)} problems × {len(models)} models × {len(variant_indices)} variants")

    pairs: List[Pair] = []
    skipped: List[str] = []

    for problem_rel in problems:
        problem_path = BASE_DIR / "problems" / problem_rel
        if validate_paths and not problem_path.exists():
            skipped.append(f"problem '{problem_rel}' not found at {problem_path}")
            continue

        try:
            problem_name = get_problem_name(Path("problems") / problem_rel)
        except ValueError as e:
            skipped.append(str(e))
            continue

        for model in models:
            model_prefix = get_model_prefix(model)

            for variant_idx in variant_indices:
                suffix = "" if variant_idx == 0 else f"_{variant_idx}"
                solution_name = f"{model_prefix}_{problem_name}{suffix}"

                if validate_paths:
                    solution_path = BASE_DIR / "solutions" / solution_name
                    if not solution_path.exists():
                        skipped.append(f"solution '{solution_name}' not found at {solution_path}")
                        continue

                pairs.append(Pair(solution=solution_name, problem=problem_rel))

    if skipped:
        logger.warning(f"Skipped {len(skipped)} invalid pairs:")
        for msg in skipped[:10]:
            logger.warning(f"  - {msg}")
        if len(skipped) > 10:
            logger.warning(f"  ... and {len(skipped) - 10} more")

    return pairs


def read_pairs(pairs_path: Path, *, validate_paths: bool = True) -> List[Pair]:
    pairs: List[Pair] = []
    skipped: List[Tuple[str, str]] = []
    with pairs_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" not in stripped:
                raise ValueError(f"Invalid pair line (expected solution:problem): {stripped}")
            solution, problem = stripped.split(":", 1)
            solution = solution.strip()
            problem = problem.strip()

            if validate_paths:
                solution_path = BASE_DIR / "solutions" / solution
                problem_path = BASE_DIR / "problems" / problem
                missing = []
                if not solution_path.exists():
                    missing.append(f"solution '{solution}' not found at {solution_path}")
                if not problem_path.exists():
                    missing.append(f"problem '{problem}' not found at {problem_path}")
                if missing:
                    for msg in missing:
                        logger.error(f"SKIP: {msg}")
                    skipped.append((solution, problem))
                    continue

            pairs.append(Pair(solution, problem))

    if skipped:
        logger.warning(f"Skipped {len(skipped)} pair(s) due to missing paths.")

    if not pairs:
        raise ValueError(f"No valid pairs found in {pairs_path}")
    return pairs


def prepare_job_workspace(
    pair: Pair,
    *,
    ordinal: int,
    total: int,
    batches_root: Path,
    cloud: str,
    region: Optional[str],
    cpus: Optional[str],
    memory: Optional[str],
    disk_size: Optional[int],
    disk_tier: Optional[str],
    accelerators: Optional[str],
    docker_image: str,
    force_recreate: bool,
    gpu_accelerator: Optional[str],
    gpu_config: dict[str, Tuple[Optional[str], bool, bool]],
) -> JobContext:
    job_dir = batches_root / pair.job_name
    if job_dir.exists() and force_recreate:
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    # Strip trailing slashes to avoid SkyPilot mount path errors
    problem_path = pair.problem.rstrip("/")
    solution_path = pair.solution.rstrip("/")

    pairs_file = job_dir / "research/pairs.txt"
    pairs_file.write_text(f"{solution_path}:{problem_path}\n", encoding="utf-8")

    remote_base = "~/sky_workdir"
    file_mounts: dict[str, str] = {}

    def mount(relative_remote: str, local: Path) -> None:
        if local.exists():
            file_mounts[f"{remote_base}/{relative_remote}"] = str(local.resolve())

    mount("main_loop.sh", BASE_DIR / "main_loop.sh")
    mount("research/docker_images.txt", BASE_DIR / "research/docker_images.txt")
    mount("entrypoint.sh", BASE_DIR / "entrypoint.sh")
    mount("include", BASE_DIR / "include")
    mount(f"research/{problem_path}", BASE_DIR / "problems" / problem_path)

    # Mount common/ directories from parent levels (for shared evaluator code)
    parts = problem_path.split("/")
    for i in range(1, len(parts)):
        parent = "/".join(parts[:i])
        common_dir = BASE_DIR / "problems" / parent / "common"
        if common_dir.is_dir():
            mount(f"research/{parent}/common", common_dir)

    mount(f"solutions/{solution_path}", BASE_DIR / "solutions" / solution_path)
    mount("research/pairs.txt", pairs_file)

    # Load problem's runtime config from config.yaml
    problem_full_path = BASE_DIR / "problems" / problem_path
    runtime_config = load_runtime_config(problem_full_path)
    res_config = runtime_config.resources

    # Determine final resource values with priority:
    # 1. CLI arguments (highest priority)
    # 2. config.yaml runtime.resources
    # 3. docker_images.txt (legacy GPU detection)
    # 4. Defaults

    accelerators_value = accelerators
    cpus_value = cpus
    memory_value = memory
    disk_size_value = disk_size
    disk_tier_value = disk_tier
    cloud_value = cloud
    region_value = region
    instance_type_value: Optional[str] = None
    image_id_value: Optional[str] = None

    # Apply config.yaml resources - config.yaml takes priority over CLI defaults
    # This allows problem-specific configs to override script defaults
    if res_config.accelerators and not accelerators:
        accelerators_value = res_config.accelerators
    if res_config.cpus and not cpus:
        cpus_value = res_config.cpus
    if res_config.memory and not memory:
        memory_value = res_config.memory
    if res_config.disk_size:
        disk_size_value = res_config.disk_size
    if res_config.disk_tier:
        disk_tier_value = res_config.disk_tier
    if res_config.cloud:
        cloud_value = res_config.cloud
    if res_config.region:
        region_value = res_config.region
    if res_config.instance_type:
        instance_type_value = res_config.instance_type
    if res_config.image_id:
        image_id_value = res_config.image_id

    # Legacy: docker_images.txt GPU detection (only if no accelerators specified yet)
    image_override: Optional[str] = None
    dind_required = False
    base_problem = problem_path.split("/", 1)[0]
    image_flag = gpu_config.get(base_problem)
    if image_flag:
        image_override, requires_gpu, dind_required = image_flag
        if not accelerators_value and requires_gpu:
            accelerators_value = gpu_accelerator

    # Always run on VM directly, let main_loop.sh handle Docker containers
    # This ensures consistent behavior: VM -> main_loop.sh -> docker run (using docker_images.txt)
    # image_id from config.yaml is respected for custom AMIs (e.g., AWS Deep Learning AMIs)
    resources = sky.Resources(
        cloud=cloud_value,  # type: ignore[arg-type]  # SkyPilot accepts string
        region=region_value,
        cpus=cpus_value,
        memory=memory_value,
        accelerators=accelerators_value,
        disk_size=disk_size_value,
        disk_tier=disk_tier_value,
        instance_type=instance_type_value,
        image_id=image_id_value,  # Custom AMI from config.yaml if specified
    )

    # Always install Docker on VM since main_loop.sh needs it
    docker_install_cmd = textwrap.dedent(
        """\
        # Install Docker for main_loop.sh
        if ! command -v docker &>/dev/null; then
            curl -fsSL https://get.docker.com | sudo sh
            sudo usermod -aG docker $USER
            sudo systemctl start docker
        fi
        """
    )

    setup_commands = textwrap.dedent(
        """\
        set -euo pipefail
        chmod +x ~/sky_workdir/main_loop.sh ~/sky_workdir/entrypoint.sh 2>/dev/null || true
        find ~/sky_workdir/problems -maxdepth 2 -type f -name '*.sh' -exec chmod +x {} \\; 2>/dev/null || true
        find ~/sky_workdir/solutions -maxdepth 2 -type f -name '*.sh' -exec chmod +x {} \\; 2>/dev/null || true
        """
    ) + docker_install_cmd

    run_commands = textwrap.dedent(
        """\
        set -euo pipefail
        cd ~/sky_workdir
        ./main_loop.sh
        """
    )

    task = sky.Task(
        name=pair.job_name,
        setup=setup_commands,
        run=run_commands,
        file_mounts=file_mounts,
    )
    task.set_resources(resources)

    cluster_name = pair.job_name
    return JobContext(
        pair=pair,
        job_dir=job_dir,
        cluster_name=cluster_name,
        task=task,
        ordinal=ordinal,
        total=total,
    )


def run_command(cmd: List[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a shell command and log its output."""
    cmd_str = " ".join(cmd)
    logger.debug(f"Running command: {cmd_str}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        if result.stdout:
            logger.error(result.stdout.strip())
        if result.stderr:
            logger.error(result.stderr.strip())
    return result


def collect_job_logs(cluster: str, destination: Path) -> Optional[Path]:
    """Collect SkyPilot logs from a cluster."""
    destination.mkdir(parents=True, exist_ok=True)
    log_path = destination / "sky_logs.txt"
    cmd = ["sky", "logs", cluster, "1"]
    result = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True)
    if result.returncode == 0 and result.stdout:
        log_path.write_text(result.stdout, encoding="utf-8")
        return log_path
    error_msg = result.stderr.strip() or result.stdout.strip()
    if error_msg:
        logger.warning(f"Failed to collect sky logs for {cluster}: {error_msg}")
    else:
        logger.warning(f"Failed to collect sky logs for {cluster}")
    return None


def teardown_context(ctx: JobContext, *, keep_cluster: bool, reuse: bool) -> None:
    """Tear down cluster and cleanup job workspace."""
    if ctx.torn_down:
        return
    if ctx.launched and not keep_cluster:
        try:
            down_request = sky.down(ctx.cluster_name)
            sky.stream_and_get(down_request)
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            logger.warning(f"Failed to tear down {ctx.cluster_name}: {exc}")
    if not reuse:
        shutil.rmtree(ctx.job_dir, ignore_errors=True)
    ctx.torn_down = True


def fetch_results(ctx: JobContext, remote_root: Path, merge_dest: Path) -> None:
    cluster = ctx.cluster_name
    job_name = ctx.pair.job_name
    remote_dir = remote_root / job_name
    if remote_dir.exists():
        shutil.rmtree(remote_dir)
    remote_dir.mkdir(parents=True, exist_ok=True)

    results_parent = remote_dir

    results_ok = scp_directory(
        cluster,
        "~/sky_workdir/results",
        results_parent,
        handle=ctx.handle,
    )
    if not results_ok:
        log_path = collect_job_logs(cluster, remote_dir)
        message = f"Result directory missing for {cluster}; aborting fetch."
        if log_path:
            message += f" See {log_path} for captured logs."
        raise RuntimeError(message)
    merge_results(remote_dir / "results", merge_dest / "results")

    # gpt_generation_logs are only created during solution generation workflows;
    # evaluation jobs generally do not produce them, so we skip syncing here to
    # avoid unnecessary warnings and delays.

@dataclass
class DirectConnectionInfo:
    user: str
    host: str
    key_path: Optional[str]
    port: int


def _load_cluster_auth_config(cluster_yaml: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not cluster_yaml:
        return None, None
    yaml_path = Path(cluster_yaml).expanduser()
    if not yaml_path.is_file():
        return None, None
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return None, None
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None, None
    auth = data.get("auth", {})
    ssh_user = auth.get("ssh_user")
    ssh_key = auth.get("ssh_private_key")
    if ssh_key:
        ssh_key = os.path.expanduser(ssh_key)
    return ssh_user, ssh_key


def _build_direct_connection(handle: Optional[object]) -> Optional[DirectConnectionInfo]:
    if handle is None:
        return None
    ssh_user = getattr(handle, "ssh_user", None)
    head_ip = getattr(handle, "head_ip", None)
    port = getattr(handle, "head_ssh_port", None) or 22
    update_ips = getattr(handle, "update_cluster_ips", None)
    if head_ip is None and callable(update_ips):
        try:
            update_ips(max_attempts=5)
        except Exception:
            pass
        head_ip = getattr(handle, "head_ip", None)
    cluster_yaml = getattr(handle, "cluster_yaml", None)
    yaml_user, yaml_key = _load_cluster_auth_config(cluster_yaml)
    if ssh_user is None:
        ssh_user = yaml_user
    ssh_key = yaml_key
    if ssh_key and not Path(ssh_key).exists():
        ssh_key = None
    if ssh_user and head_ip:
        return DirectConnectionInfo(user=ssh_user, host=head_ip, key_path=ssh_key, port=port)
    return None


def scp_directory(
    cluster: str,
    remote_path: str,
    local_parent: Path,
    retries: int = 12,  # Ignored but kept for backwards-compat keyword compatibility.
    delay_seconds: float = 10.0,  # Ignored but kept for backwards-compat keyword compatibility.
    *,
    handle: Optional[object] = None,
) -> bool:
    local_parent.mkdir(parents=True, exist_ok=True)
    dest = local_parent
    ssh_options = [
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "LogLevel=ERROR",
    ]
    direct_conn = _build_direct_connection(handle)
    use_targets: List[Tuple[str, List[str]]] = []
    # Primary attempt via cluster alias.
    use_targets.append((cluster, []))

    def _direct_target() -> Optional[Tuple[str, List[str]]]:
        if direct_conn is None:
            return None
        identity_args: List[str] = []
        if direct_conn.key_path:
            identity_args.extend(["-i", direct_conn.key_path])
        if direct_conn.port and direct_conn.port != 22:
            identity_args.extend(["-P", str(direct_conn.port)])
        target = f"{direct_conn.user}@{direct_conn.host}"
        return target, identity_args

    direct_target = _direct_target()
    if direct_target is not None:
        use_targets.append(direct_target)

    for target, extra_opts in use_targets:
        cmd = [
            "scp",
            *ssh_options,
            *extra_opts,
            "-r",
            f"{target}:{remote_path}",
            str(dest),
        ]
        result = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True)
        if result.returncode == 0:
            return True
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        details = stderr or stdout
        logger.warning(
            f"scp failed for {remote_path} from {target}: {details or 'no details'}"
        )

    return False


def merge_results(src_results: Path, dest_results: Path) -> None:
    if not src_results.exists():
        return

    merge_tree_filtered(src_results, dest_results, exclude_files={"summary.txt", "results.csv"})

    src_summary = src_results / "summary.txt"
    dest_summary = dest_results / "summary.txt"
    if src_summary.exists():
        append_with_header(src_summary, dest_summary, label=src_results.parent.name)

    src_csv = src_results / "results.csv"
    dest_csv = dest_results / "results.csv"
    if src_csv.exists():
        append_csv(src_csv, dest_csv)


def merge_tree(src: Path, dest: Path) -> None:
    merge_tree_filtered(src, dest, exclude_files=set())


def merge_tree_filtered(src: Path, dest: Path, *, exclude_files: set[str]) -> None:
    if not src.exists():
        return
    for path in src.rglob("*"):
        if path.is_dir():
            continue
        if path.name in exclude_files:
            continue
        rel = path.relative_to(src)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def append_with_header(src: Path, dest: Path, *, label: Optional[str] = None) -> None:
    content = src.read_text(encoding="utf-8")
    label_text = label or src.parent.name
    header = f"\n\n==== Imported from {label_text} ===="
    if dest.exists():
        with dest.open("a", encoding="utf-8") as f:
            f.write(header)
            f.write("\n")
            f.write(content)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")


def append_csv(src: Path, dest: Path) -> None:
    lines = src.read_text(encoding="utf-8").splitlines()
    if not lines:
        return
    header, *rows = lines
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_header = not dest.exists()
    with dest.open("a", encoding="utf-8") as f:
        if write_header:
            f.write(header + "\n")
        for row in rows:
            if row.strip():
                f.write(row + "\n")


def cancel_inflight_requests(
    contexts: Iterable[JobContext],
    *,
    keep_cluster: bool,
    reuse: bool,
) -> None:
    """Cancel in-flight SkyPilot requests and clean up resources."""
    request_ids = [ctx.request_id for ctx in contexts if ctx.request_id]
    if request_ids:
        try:
            cancel_request = sky.api_cancel(request_ids, silent=True)
            sky.stream_and_get(cancel_request)
        except Exception as exc:  # pragma: no cover - best-effort cancel
            logger.warning(f"Failed to cancel SkyPilot requests: {exc}")

    for ctx in contexts:
        if ctx.launched:
            try:
                teardown_context(ctx, keep_cluster=keep_cluster, reuse=reuse)
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                logger.warning(f"Failed to tear down {ctx.cluster_name}: {exc}")


def execute_job_context(
    ctx: JobContext,
    *,
    skip_download: bool,
    keep_cluster: bool,
    reuse: bool,
    remote_root: Path,
    merge_dest: Path,
) -> Tuple[JobContext, int, Optional[Exception]]:
    retcode = 0
    error: Optional[Exception] = None
    buffer = io.StringIO()

    try:
        format_progress(ctx, "LAUNCH", f"{ctx.pair.job_name} (cluster={ctx.cluster_name})")
        with redirect_stdout(buffer), redirect_stderr(buffer):
            request_id = sky.launch(ctx.task, cluster_name=ctx.cluster_name)
        ctx.launched = True
        ctx.request_id = str(request_id)
        format_progress(ctx, "REQUEST", ctx.request_id or "")

        with redirect_stdout(buffer), redirect_stderr(buffer):
            result = sky.stream_and_get(request_id)
        job_id: Optional[int] = None
        handle: Optional[object] = None
        if isinstance(result, tuple):
            if len(result) > 0:
                job_id = result[0]
            if len(result) > 1:
                handle = result[1]
        ctx.handle = handle
        format_progress(ctx, "REQUEST", f"{ctx.pair.job_name} submitted (job_id={job_id})")

        # Block until the managed job completes by tailing logs.
        exit_code = 0
        if job_id is not None:
            with redirect_stdout(buffer), redirect_stderr(buffer):
                exit_code = sky.tail_logs(ctx.cluster_name, job_id, follow=True)  # returns 0 on success
        format_progress(ctx, "COMPLETE", f"{ctx.pair.job_name} (job_id={job_id})")

        if not skip_download and exit_code == 0:
            fetch_results(ctx, remote_root, merge_dest)
            format_progress(ctx, "SYNC", ctx.cluster_name)
        elif not skip_download and exit_code != 0:
            raise RuntimeError(f"Remote job failed with exit code {exit_code}")
    except Exception as exc:  # broad to surface any job failure
        error = exc
        retcode = 1
        format_progress(ctx, "ERROR", f"{ctx.pair.job_name}: {exc}")
        output = buffer.getvalue().strip()
        if output:
            for line in output.splitlines():
                format_progress(ctx, "DETAIL", line)
    finally:
        try:
            teardown_context(ctx, keep_cluster=keep_cluster, reuse=reuse)
        except Exception as teardown_exc:  # pragma: no cover - best-effort cleanup
            logger.warning(f"Failed to tear down {ctx.cluster_name}: {teardown_exc}")

    return ctx, retcode, error


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Launch per-solution SkyPilot jobs.")
    # Input source: either pairs file OR problem list (expands using models.txt)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--pairs-file", help="Pairs file to read (solution:problem format)")
    input_group.add_argument("--problem-list", help="Problem list file (auto-expands with models.txt and num_solutions.txt)")
    parser.add_argument("--models-file", default="research/models.txt", help="Models file for --problem-list expansion (default: models.txt)")
    parser.add_argument("--num-solutions-file", default="research/num_solutions.txt", help="Variant indices file (default: num_solutions.txt)")
    parser.add_argument("--batches-root", default="skypilot_batches", help="Directory to store per-solution workspaces")
    parser.add_argument("--remote-root", default="remote_results", help="Directory to receive downloaded artifacts")
    parser.add_argument("--max-concurrent", type=int, default=20, help="Maximum concurrent launches handled locally")
    parser.add_argument("--cloud", default="gcp", help="Target cloud provider (default: gcp)")
    parser.add_argument("--gcp", action="store_true", help="Use GCP (overrides --cloud to gcp)")
    parser.add_argument("--region", help="Cloud region (optional)")
    parser.add_argument("--cpus", default="8+", help="CPU requirement (default: 8+)")
    parser.add_argument("--memory", default="16+", help="Memory requirement (default: 16+)")
    parser.add_argument("--disk-size", type=int, default=50, help="Disk size in GB (default: 50)")
    parser.add_argument("--disk-tier", default="low", help="Disk tier: low, medium, high, best, ultra (default: low)")
    parser.add_argument("--accelerators", help="Accelerator specification, e.g. L4:1 (optional)")
    parser.add_argument(
        "--gpu-accelerator",
        default="L4:1",
        help="Accelerator spec to use when a problem requires GPU (default: L4:1)",
    )
    parser.add_argument("--docker-image", default="python:3.11-slim-trixie", help="Docker image used in main_loop (default matches repo)")
    parser.add_argument("--reuse", action="store_true", help="Reuse existing batch directories instead of recreating")
    parser.add_argument("--skip-download", action="store_true", help="Skip rsync/download after launch (debug)")
    parser.add_argument("--keep-cluster", action="store_true", help="Do not tear down cluster after completion")
    parser.add_argument("--dry-run", action="store_true", help="Prepare workspaces but do not run sky launch")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation of solution/problem paths")
    args = parser.parse_args(argv)

    if args.max_concurrent <= 0:
        parser.error("--max-concurrent must be a positive integer.")

    # Override cloud to GCP if --gcp flag is set
    if args.gcp:
        args.cloud = "gcp"

    # Determine input source and load pairs
    if args.problem_list:
        # Expand from problem list using models.txt and num_solutions.txt
        problem_list_path = BASE_DIR / args.problem_list
        if not problem_list_path.exists():
            parser.error(f"Problem list file not found: {problem_list_path}")
        models_path = BASE_DIR / args.models_file
        num_solutions_path = BASE_DIR / args.num_solutions_file
        pairs = expand_pairs_from_problems(
            problem_list_path,
            models_path,
            num_solutions_path,
            validate_paths=not args.skip_validation
        )
        logger.info(f"Expanded to {len(pairs)} pairs from {args.problem_list}")
    else:
        # Use pairs file (default to pairs.txt if nothing specified)
        pairs_file = args.pairs_file or "research/pairs.txt"
        pairs_path = BASE_DIR / pairs_file
        if not pairs_path.exists():
            parser.error(f"Pairs file not found: {pairs_path}")
        pairs = read_pairs(pairs_path, validate_paths=not args.skip_validation)
    batches_root = (BASE_DIR / args.batches_root).resolve()
    remote_root = (BASE_DIR / args.remote_root).resolve()
    merge_dest = BASE_DIR.resolve()

    docker_gpu_config = load_docker_gpu_config(BASE_DIR / "research/docker_images.txt")

    total_pairs = len(pairs)
    job_contexts: List[JobContext] = []
    for idx, pair in enumerate(pairs, start=1):
        ctx = prepare_job_workspace(
            pair,
            ordinal=idx,
            total=total_pairs,
            batches_root=batches_root,
            cloud=args.cloud,
            region=args.region,
            cpus=args.cpus,
            memory=args.memory,
            disk_size=args.disk_size,
            disk_tier=args.disk_tier,
            accelerators=args.accelerators,
            docker_image=args.docker_image,
            force_recreate=not args.reuse,
            gpu_accelerator=args.gpu_accelerator,
            gpu_config=docker_gpu_config,
        )
        job_contexts.append(ctx)
        format_progress(ctx, "QUEUE", ctx.pair.job_name)

    if args.dry_run:
        logger.info("Workspaces prepared. Exiting before launch (dry-run mode).")
        return 0

    failures: List[Tuple[Pair, Optional[Exception]]] = []
    global progress_bar
    progress_bar = tqdm(total=total_pairs, desc="Evaluations", unit="pair", dynamic_ncols=True)

    with ThreadPoolExecutor(max_workers=args.max_concurrent) as executor:
        futures = {
            executor.submit(
                execute_job_context,
                ctx,
                skip_download=args.skip_download,
                keep_cluster=args.keep_cluster,
                reuse=args.reuse,
                remote_root=remote_root,
                merge_dest=merge_dest,
            ): ctx
            for ctx in job_contexts
        }

        try:
            for future in as_completed(futures):
                ctx, retcode, error = future.result()
                if progress_bar is not None:
                    progress_bar.update(1)
                if retcode != 0:
                    failures.append((ctx.pair, error))
        except KeyboardInterrupt:
            logger.warning("\nCtrl-C received; cancelling outstanding jobs...")
            for future in futures:
                future.cancel()
            cancel_ids = [ctx.request_id for ctx in job_contexts if ctx.request_id]
            if cancel_ids:
                try:
                    cancel_request = sky.api_cancel(cancel_ids, silent=True)
                    sky.stream_and_get(cancel_request)
                except Exception as exc:  # pragma: no cover - best-effort
                    logger.warning(f"Failed to cancel running SkyPilot requests: {exc}")
            for ctx in job_contexts:
                if ctx.launched:
                    teardown_context(ctx, keep_cluster=args.keep_cluster, reuse=args.reuse)
            if progress_bar is not None:
                progress_bar.close()
                progress_bar = None
            return 130

        # Drain any futures that may still hold exceptions to avoid thread leaks.
        for future in futures:
            if future.done():
                continue
            try:
                future.result()
            except Exception:
                pass

    if progress_bar is not None:
        progress_bar.close()
        progress_bar = None

    if failures:
        print("\n=== Summary ===")
        print(f"Failed jobs: {len(failures)}/{total_pairs}")
        for pair, error in failures:
            details = f": {error}" if error else ""
            logger.error(f"Failed: {pair.solution}:{pair.problem}{details}")
        return 1

    print("\n=== Summary ===")
    print(f"All {total_pairs} jobs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
