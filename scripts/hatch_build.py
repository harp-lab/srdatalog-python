'''Hatchling build hook: ensure `runtime/vendor/` is populated before
the wheel is assembled.

Wired via `pyproject.toml`:

    [tool.hatch.build.hooks.custom]
    path = "scripts/hatch_build.py"

`hatch build` (or `uv build`) will instantiate `VendorBuildHook` and
call `initialize()` once per build target. We only act on `wheel`
builds — sdist ships without vendor (that's the point: small source
distribution, fat binary wheel).
'''
from __future__ import annotations

import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class VendorBuildHook(BuildHookInterface):
  '''Populate runtime/vendor/ before wheel assembly.'''
  PLUGIN_NAME = "srdatalog-vendor"

  def initialize(self, version: str, build_data: dict) -> None:
    if self.target_name != "wheel":
      return  # sdist, app, etc. don't need vendor

    # Make scripts/ importable so we can call populate_vendor directly
    # without a subprocess (faster, surfaces tracebacks naturally).
    scripts_dir = Path(self.root) / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
      import populate_vendor
    finally:
      sys.path.pop(0)

    # Skip the work if vendor is already at the pinned versions —
    # contributors will have populated it once after `git clone`,
    # subsequent `uv build` invocations are then no-ops.
    if populate_vendor.check() != 0:
      print("[hatch-build] populating vendor/...")
      populate_vendor.populate()
    else:
      print("[hatch-build] vendor/ already current")

    # Ensure the wheel's force-include picks up the (now-present) tree.
    build_data.setdefault("force_include", {})
    build_data["force_include"]["src/srdatalog/runtime/vendor"] = (
      "srdatalog/runtime/vendor"
    )
