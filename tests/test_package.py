# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""Smoke tests: the package installs and imports under the `mt` namespace."""

import re

import mt
from mt import geodesicdome


def test_version_is_pep440():
    assert re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?(\.dev\d+)?", geodesicdome.__version__)


def test_mt_is_a_namespace_package():
    # PEP 420 namespace packages have no __init__.py, hence no __file__.
    assert getattr(mt, "__file__", None) is None
