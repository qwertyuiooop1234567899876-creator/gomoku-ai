"""Compatibility script entry point for :mod:`tools.yixin_smoke_test`."""

from __future__ import annotations

import sys

from tools import yixin_smoke_test as _implementation


if __name__ == "__main__":
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
