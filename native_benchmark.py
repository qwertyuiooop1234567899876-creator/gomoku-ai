"""Compatibility script entry point for :mod:`tools.native_benchmark`."""

from __future__ import annotations

import sys

from tools import native_benchmark as _implementation


if __name__ == "__main__":
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
