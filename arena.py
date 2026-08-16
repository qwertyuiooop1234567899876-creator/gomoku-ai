"""Compatibility import and script entry point for :mod:`app.arena`."""

from __future__ import annotations

import sys

from app import arena as _implementation


if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
