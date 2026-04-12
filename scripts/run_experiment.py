from __future__ import annotations

try:
    from ._bootstrap import ensure_repo_root_on_path
except ImportError:
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from deepplanning.orchestration import main, run

__all__ = ["main", "run"]


if __name__ == "__main__":
    main()
