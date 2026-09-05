'''Opt-in CUDA plus PyReason parity test for the checked-in minimal example.'''

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
  os.environ.get("SRDATALOG_RUN_VULREASONER_PARITY") != "1",
  reason="set SRDATALOG_RUN_VULREASONER_PARITY=1 for the GPU/PyReason integration",
)


def test_real_graphml_temporal_analyst_rows_match_pyreason():
  root = Path(__file__).resolve().parents[1]
  command = [
    sys.executable,
    str(root / "minimal_vulreasoner" / "validate_minimal_parity.py"),
  ]
  if os.environ.get("SRDATALOG_VULREASONER_NO_COMPILE") == "1":
    command.append("--no-compile")
  completed = subprocess.run(
    command,
    cwd=root,
    text=True,
    capture_output=True,
    check=False,
  )
  assert completed.returncode == 0, completed.stdout + completed.stderr
  assert "PASS: complete temporal AnalystAt rows match PyReason" in completed.stdout
