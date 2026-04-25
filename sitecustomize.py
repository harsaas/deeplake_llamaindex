"""Startup workaround for a llama-index import crash on Python 3.10.

Background:
- Some llama-index-core 0.10.x builds annotate parameters as Optional[TypeAlias].
- typing.TypeAlias is a special form, not a real type, so Optional[TypeAlias]
  raises a TypeError at import time on Python 3.10.

This file is auto-imported by Python (if on sys.path) and applies a very narrow
workaround only when running:
- Python 3.10
- llama-index-core 0.10.x

If/when you upgrade llama-index-core beyond 0.10.x, this becomes a no-op.
"""

from __future__ import annotations

import sys
import typing


def _workaround_llama_index_typealias_crash() -> None:
    if sys.version_info[:2] != (3, 10):
        return

    type_alias = getattr(typing, "TypeAlias", None)
    if type_alias is None:
        return

    # Probe whether TypeAlias is rejected by this interpreter when used as a
    # type argument (this breaks certain llama-index-core 0.10.x builds at import time).
    try:
        typing.Optional[type_alias]  # type: ignore[index]
        typing.List[type_alias]  # type: ignore[index]
        typing.Sequence[type_alias]  # type: ignore[index]
    except TypeError:
        # Minimal impact: only rewrite TypeAlias to a usable type.
        typing.TypeAlias = typing.Any  # type: ignore[attr-defined]


_workaround_llama_index_typealias_crash()
