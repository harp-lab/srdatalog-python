'''S4.7 — RawString count discipline (whole-tree).

Per docs/stage4_iir_vocabulary.md §2 + docs/ir_dialect_contract.md §4:
`iir.cf.RawString` is a transition-only escape hatch. After Stage 4
close-out, *no* production lowering or codegen module constructs
`RawString(...)` — `iir.cf.UserCode` covers user-supplied expression
text (Category J), and structured ops cover everything else.

This test enforces that across the whole `src/srdatalog/ir/` tree, not
just `sorted_array/lowerings.py` (the original Stage 4 target). Any
new `RawString(...)` call site anywhere in the IR layer is a contract
violation — either:

  (a) Use the existing `iir.cf.UserCode(text)` op for user-supplied
      C++ fragments (Filter/ConstantBind code).
  (b) Add a structured op to the appropriate dialect.
  (c) If you genuinely need a new escape hatch, motivate it in the
      PR description AND update _MAX_RAWSTRING_CALLS below in the same
      commit (the strict-pin test forces atomic updates).

The cap below is the RATCHET — only DECREASE it. Increases must be
explicitly motivated.
'''

from __future__ import annotations

from pathlib import Path

# The whole IR tree. Excludes the class definition + tests.
_IR_ROOT = Path(__file__).resolve().parent.parent / 'src' / 'srdatalog' / 'ir'

# Files that LEGITIMATELY contain `RawString(` and must be excluded
# from the cap check:
#   - `dialects/iir/cf/ops.py` defines `class RawString(Op):`.
_EXCLUDED = {
  _IR_ROOT / 'dialects' / 'iir' / 'cf' / 'ops.py',
}


def _all_ir_call_sites() -> dict[Path, int]:
  '''Return {path: count} for every file under src/srdatalog/ir/ that
  contains `RawString(` calls (excluding the class definition site).'''
  out: dict[Path, int] = {}
  for p in _IR_ROOT.rglob('*.py'):
    if p in _EXCLUDED:
      continue
    text = p.read_text()
    n = text.count('RawString(')
    if n > 0:
      out[p] = n
  return out


# RATCHET: only DECREASE this. New uses must motivate the increase
# explicitly in the PR description + this docstring + the per-file
# inventory below.
_MAX_RAWSTRING_CALLS = 0


def test_no_raw_string_calls_in_production_ir() -> None:
  '''After Stage 4 close-out, no production IR module constructs
  RawString. Any non-zero count anywhere is a regression.'''
  sites = _all_ir_call_sites()
  total = sum(sites.values())

  if total > _MAX_RAWSTRING_CALLS:
    breakdown = '\n'.join(
      f'  {p.relative_to(_IR_ROOT.parent.parent.parent)}: {n}' for p, n in sorted(sites.items())
    )
    raise AssertionError(
      f'RawString(...) call sites across src/srdatalog/ir/: {total}, '
      f'cap is {_MAX_RAWSTRING_CALLS}.\n'
      f'Breakdown:\n{breakdown}\n\n'
      'New RawString uses must either:\n'
      '  (a) Use iir.cf.UserCode for user-supplied C++ fragments.\n'
      '  (b) Use a structured op from the appropriate dialect.\n'
      '  (c) Document as a new Category in stage4_iir_vocabulary.md\n'
      '      AND raise _MAX_RAWSTRING_CALLS in this file in the\n'
      '      same commit.\n'
      'See docs/ir_dialect_contract.md §4 for the contract.'
    )


def test_raw_string_count_pinned_exact() -> None:
  '''If RawString usage DROPS below the cap, update
  _MAX_RAWSTRING_CALLS in the same commit. This test forces the
  ratchet to be atomic — no silent slack.'''
  total = sum(_all_ir_call_sites().values())

  assert total == _MAX_RAWSTRING_CALLS, (
    f'RawString call total is {total}; the cap '
    f'_MAX_RAWSTRING_CALLS is {_MAX_RAWSTRING_CALLS}. If you removed '
    f'a RawString use, update the cap to {total} in the same commit. '
    'This test is intentionally strict — the ratchet is monotonically '
    'decreasing.'
  )
