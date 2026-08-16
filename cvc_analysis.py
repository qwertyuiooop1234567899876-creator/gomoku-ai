"""Compatibility import and script entry point for :mod:`tools.cvc_analysis`."""

from __future__ import annotations

import sys

from tools import cvc_analysis as _implementation


if __name__ == "__main__":
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
