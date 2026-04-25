"""Shim to ensure the repo-level `sitecustomize.py` is applied.

When you run a script via `python scripts/foo.py`, Python sets `sys.path[0]`
(the first import location) to the `scripts/` directory.

Python's `site` module will auto-import a module named `sitecustomize` *only*
if it can be found on `sys.path` at startup. Since the repo root is not on
`sys.path` in this execution mode, the top-level `sitecustomize.py` would not
be discovered.

This shim is intentionally tiny: it loads and executes the repo-level
`sitecustomize.py` under a different module name.
"""

from __future__ import annotations

from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path


def _load_repo_sitecustomize() -> None:
    repo_sitecustomize = Path(__file__).resolve().parents[1] / "sitecustomize.py"
    if not repo_sitecustomize.exists():
        return

    loader = SourceFileLoader("_repo_sitecustomize", str(repo_sitecustomize))
    spec = spec_from_loader(loader.name, loader)
    if spec is None:
        return

    module = module_from_spec(spec)
    loader.exec_module(module)


_load_repo_sitecustomize()
