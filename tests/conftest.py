'''pytest conftest — adds the tests/ directory to `sys.path` so
test-to-test cross-imports keep working (e.g.,
`from test_integration_triangle import build_triangle`).

The package under test (`srdatalog`) is resolved via the installed
source tree (`src/srdatalog/`) through hatchling's editable-install.
Tests refer to it as `from srdatalog.xxx import ...`.
'''
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
