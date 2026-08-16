"""Compatibility import and script entry point for manual scenarios."""

from __future__ import annotations

import sys

from tools import manual_scenarios as _implementation


if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
