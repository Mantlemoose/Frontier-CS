#!/usr/bin/env python3
"""
Shared configuration loader for problem runtime settings.

This module provides utilities to load and parse problem config.yaml files,
especially the runtime.resources section which mirrors SkyPilot's resources API.

Example config.yaml:
```yaml
{
  "runtime": {
    "timeout_seconds": 3600,
    "resources": {
      "accelerators": "L4:1",
      "instance_type": "n1-standard-8",
      "cpus": "8+",
      "memory": "32+",
      "disk_size": 100
    }
  }
}
```
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ResourcesConfig:
    """
    SkyPilot-compatible resources configuration.

    Mirrors sky.Resources parameters that can be specified in config.yaml.
    All fields are optional - unspecified fields use SkyPilot/CLI defaults.
    """
    # GPU/Accelerator specification (e.g., "L4:1", "A100:4", "H100:8")
    accelerators: Optional[str] = None

    # Cloud-specific instance type (e.g., "n1-standard-8", "p3.2xlarge")
    instance_type: Optional[str] = None

    # CPU specification (e.g., "8", "8+", "4-8")
    cpus: Optional[str] = None

    # Memory specification in GB (e.g., "32", "32+", "16-64")
    memory: Optional[str] = None

    # Disk size in GB
    disk_size: Optional[int] = None

    # Disk tier (e.g., "high", "medium", "low")
    disk_tier: Optional[str] = None

    # Preferred cloud provider (e.g., "gcp", "aws", "azure")
    cloud: Optional[str] = None

    # Preferred region
    region: Optional[str] = None

    # Custom image ID (for Docker or VM images)
    image_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, excluding None values."""
        return {k: v for k, v in {
            "accelerators": self.accelerators,
            "instance_type": self.instance_type,
            "cpus": self.cpus,
            "memory": self.memory,
            "disk_size": self.disk_size,
            "disk_tier": self.disk_tier,
            "cloud": self.cloud,
            "region": self.region,
            "image_id": self.image_id,
        }.items() if v is not None}

    @property
    def has_gpu(self) -> bool:
        """Check if this config specifies GPU resources."""
        return self.accelerators is not None

    @property
    def gpu_type(self) -> Optional[str]:
        """Extract GPU type from accelerators string (e.g., 'L4:1' -> 'L4')."""
        if not self.accelerators:
            return None
        return self.accelerators.split(":")[0]


@dataclass
class RuntimeConfig:
    """
    Complete runtime configuration from config.yaml.
    """
    # Timeout in seconds for evaluation
    timeout_seconds: Optional[int] = None

    # Whether GPU is required (legacy field, prefer resources.accelerators)
    requires_gpu: Optional[bool] = None

    # SkyPilot-compatible resources configuration
    resources: ResourcesConfig = field(default_factory=ResourcesConfig)

    # Problem-specific environment description (for LLM prompts)
    environment: Optional[str] = None

    # Legacy gpu_type field (prefer resources.accelerators)
    gpu_type: Optional[str] = None


def load_runtime_config(problem_path: Path) -> RuntimeConfig:
    """
    Load runtime configuration from problem's config.yaml.

    Args:
        problem_path: Path to the problem directory

    Returns:
        RuntimeConfig with parsed values (defaults for missing fields)
    """
    config_file = problem_path / "config.yaml"
    runtime_config = RuntimeConfig()

    if not config_file.exists():
        return runtime_config

    try:
        config = json.loads(config_file.read_text())
    except (json.JSONDecodeError, Exception):
        return runtime_config

    runtime = config.get("runtime", {})

    # Parse basic runtime fields
    if runtime.get("timeout_seconds"):
        runtime_config.timeout_seconds = int(runtime["timeout_seconds"])
    if runtime.get("requires_gpu") is not None:
        runtime_config.requires_gpu = bool(runtime["requires_gpu"])
    if runtime.get("environment"):
        runtime_config.environment = str(runtime["environment"])
    if runtime.get("gpu_type"):
        runtime_config.gpu_type = str(runtime["gpu_type"])

    # Parse resources section
    resources = runtime.get("resources", {})
    if resources:
        res = runtime_config.resources
        if resources.get("accelerators"):
            res.accelerators = str(resources["accelerators"])
        if resources.get("instance_type"):
            res.instance_type = str(resources["instance_type"])
        if resources.get("cpus"):
            res.cpus = str(resources["cpus"])
        if resources.get("memory"):
            res.memory = str(resources["memory"])
        if resources.get("disk_size"):
            res.disk_size = int(resources["disk_size"])
        if resources.get("disk_tier"):
            res.disk_tier = str(resources["disk_tier"])
        if resources.get("cloud"):
            res.cloud = str(resources["cloud"])
        if resources.get("region"):
            res.region = str(resources["region"])
        if resources.get("image_id"):
            res.image_id = str(resources["image_id"])

    # Legacy compatibility: if gpu_type is set but resources.accelerators is not,
    # convert gpu_type to accelerators format
    if runtime_config.gpu_type and not runtime_config.resources.accelerators:
        runtime_config.resources.accelerators = f"{runtime_config.gpu_type}:1"

    return runtime_config


def get_effective_gpu_type(runtime_config: RuntimeConfig) -> Optional[str]:
    """
    Get effective GPU type from runtime config.

    Priority:
    1. resources.accelerators (extract type)
    2. Legacy gpu_type field
    3. None (CPU only)
    """
    if runtime_config.resources.accelerators:
        return runtime_config.resources.gpu_type
    if runtime_config.gpu_type:
        return runtime_config.gpu_type
    return None
