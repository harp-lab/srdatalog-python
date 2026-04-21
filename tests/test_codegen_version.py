'''Version enum extensions for C++ codegen.

Checks the `.method`, `.number`, `.code` properties added to
`hir_types.Version`. `.value` and `.name` must still be `"FULL"` etc.
so the MIR S-expr byte-match against Nim keeps working.
'''
import sys
from pathlib import Path


from srdatalog.hir_types import Version


def test_version_value_unchanged():
  assert Version.FULL.value == "FULL"
  assert Version.DELTA.value == "DELTA"
  assert Version.NEW.value == "NEW"


def test_version_name_unchanged():
  assert Version.FULL.name == "FULL"
  assert Version.DELTA.name == "DELTA"
  assert Version.NEW.name == "NEW"


def test_version_method():
  assert Version.FULL.method == "full"
  assert Version.DELTA.method == "delta"
  assert Version.NEW.method == "newt"


def test_version_number():
  assert Version.FULL.number == "0"
  assert Version.DELTA.number == "1"
  assert Version.NEW.number == "2"


def test_version_code():
  assert Version.FULL.code == "FULL_VER"
  assert Version.DELTA.code == "DELTA_VER"
  assert Version.NEW.code == "NEW_VER"


if __name__ == "__main__":
  test_version_value_unchanged()
  test_version_name_unchanged()
  test_version_method()
  test_version_number()
  test_version_code()
  print("OK")
