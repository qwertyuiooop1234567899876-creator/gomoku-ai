"""Compatibility import for :mod:`app.ui_common`."""

from __future__ import annotations

import sys

from app import ui_common as _implementation


sys.modules[__name__] = _implementation
