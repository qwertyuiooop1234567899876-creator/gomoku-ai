"""Compatibility script entry point for :mod:`tools.search_benchmark`."""

from __future__ import annotations

import sys

from tools import search_benchmark as _implementation


if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
