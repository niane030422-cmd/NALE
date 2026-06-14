from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def repo_dir(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def repo_path(*parts: str) -> str:
    return str(repo_dir(*parts))


def repo_path_from_env(env_var: str, *default_parts: str) -> str:
    return os.getenv(env_var, repo_path(*default_parts))


def model_path_from_env(env_var: str, *default_parts: str) -> str:
    return repo_path_from_env(env_var, *default_parts)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if value:
        return value
    raise RuntimeError(
        f"Missing environment variable: {name}. "
        f"Set {name} before running this public version."
    )
