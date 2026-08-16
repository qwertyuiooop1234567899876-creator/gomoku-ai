"""Compatibility import and script entry point for :mod:`app.web_ui`."""

from __future__ import annotations

import sys

from app import web_ui as _implementation


if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
