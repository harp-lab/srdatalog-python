'''Sphinx config for the srdatalog docs site.

Build locally with:

    uv sync --group docs
    uv run sphinx-build -W --keep-going -b html docs docs/_build/html

CI builds the same command and deploys to GitHub Pages — see
.github/workflows/docs.yml.
'''
from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable from the docs build (used by autodoc2).
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

# ---------------------------------------------------------------------------
# Project info
# ---------------------------------------------------------------------------
project = "srdatalog"
author = "SRDatalog contributors"
copyright = "2026, SRDatalog contributors"
# Keep in sync with pyproject.toml [project] version; `importlib.metadata`
# is the least-brittle way once the package is installed, but falls back
# to a hard-coded string when we're building from source without install.
try:
  from importlib.metadata import version as _pkg_version
  release = _pkg_version("srdatalog")
except Exception:
  release = "0.1.0"
version = ".".join(release.split(".")[:2])

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
  "myst_parser",              # Markdown (.md) as first-class input
  "autodoc2",                 # Renders API reference from the package
  "sphinx.ext.intersphinx",   # Link to Python / numpy / ctypes docs
  "sphinx.ext.viewcode",      # "View source" links on API pages
  "sphinx_copybutton",        # Copy-to-clipboard buttons on code blocks
]

# MyST: enable the markdown extensions we actually want.
myst_enable_extensions = [
  "colon_fence",      # ::: directives in addition to ```
  "deflist",          # definition lists
  "tasklist",         # - [x] style checklists
  # `linkify` (auto-link bare URLs) would also be useful but it
  # requires linkify-it-py which complicates the CI install; bare
  # URLs inside explicit <angle brackets> render fine without it.
]
myst_heading_anchors = 3

# autodoc2 — scans the whole srdatalog package and emits one .rst per module.
autodoc2_packages = [
  {
    "path": "../src/srdatalog",
    "auto_mode": True,
  },
]
# Keep the generated tree under docs/api/ so it's easy to .gitignore.
autodoc2_output_dir = "api"
autodoc2_render_plugin = "myst"   # emit Markdown so our theme is consistent
autodoc2_hidden_objects = {"inherited", "private"}
autodoc2_module_all_regexes = [
  # Don't walk into cache / build artifacts if any slipped in.
  r"srdatalog\.runtime\.vendor.*",
  r"srdatalog\.runtime\.generalized_datalog.*",
]
# Show type hints inline with argument names — MyST-friendly rendering.
autodoc2_sort_names = True

# Intersphinx: link to upstream stdlib docs for cross-references.
intersphinx_mapping = {
  "python": ("https://docs.python.org/3", None),
}

# ---------------------------------------------------------------------------
# HTML theme
# ---------------------------------------------------------------------------
html_theme = "furo"
html_title = "srdatalog"
html_static_path = ["_static"]
# Furo has good defaults; just nudge the sidebar to surface the API.
html_theme_options = {
  "sidebar_hide_name": False,
  "navigation_with_keys": True,
  "source_repository": "https://github.com/harp-lab/srdatalog-python",
  "source_branch": "main",
  "source_directory": "docs/",
}

# ---------------------------------------------------------------------------
# Source file types
# ---------------------------------------------------------------------------
source_suffix = {
  ".md": "markdown",
  ".rst": "restructuredtext",
}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Fail the build on warnings — matches what CI runs.
nitpicky = False
