'''S4.7 — RawString count discipline.

Per docs/stage4_iir_vocabulary.md §2 task S4.7: cap the proliferation
of `RawString(text=...)` sites in dialect lowerings. New uses must
either get structured ops (the systematic Stage 4 cleanup) or
explicit "Category J — user-supplied code" treatment.

The number below ratchets DOWN as Stage 4 continues. Don't raise it
without a documented reason — that's the regression this test catches.

Current acceptable counts:
  - sorted_array/lowerings.py: 17 (after L1 cluster cleanup in S4.6a)
    - 3 are Category J (Filter/ConstantBind user-supplied code at
      lines around 1070, 1074, 1078)
    - 2 are Category K (handle-alias inline decls with trailing
      comments — deferred per the inventory)
    - ~12 are Category L2 (dedup-table emission block — the hardest
      tier, scheduled for S4.6b)

Subsequent S4 tasks should LOWER this number. When the count drops,
update the constant below in the SAME commit that does the cleanup —
this is a contract, not a moving target.
'''

from __future__ import annotations

from pathlib import Path

# The single source-of-truth file Stage 4 targets.
_LOWERINGS_PATH = (
  Path(__file__).resolve().parent.parent
  / 'src'
  / 'srdatalog'
  / 'ir'
  / 'dialects'
  / 'relation'
  / 'sorted_array'
  / 'lowerings.py'
)

# RATCHET: only DECREASE this. New uses must motivate the increase
# explicitly in the PR description + this docstring + a Category J/K/L
# inventory note.
_MAX_RAWSTRING_CALLS = 17


def test_lowerings_rawstring_count_at_or_below_cap() -> None:
  source = _LOWERINGS_PATH.read_text()
  count = source.count('RawString(')
  # The import line `RawString,` in the iir.cf import block doesn't
  # have an opening paren after, so it's excluded by the `(` match.
  # Match the actual call sites only.

  assert count <= _MAX_RAWSTRING_CALLS, (
    f'RawString(...) call count in sorted_array/lowerings.py is {count}, '
    f'cap is {_MAX_RAWSTRING_CALLS}. New RawString uses must either:\n'
    f'  (a) Replace an existing one (net-neutral) and update _MAX_RAWSTRING_CALLS\n'
    f'  (b) Document as Category J (user-supplied code) in the Stage 4 inventory\n'
    f'  (c) Be replaced with a structured IIR op per the Stage 4 vocabulary plan\n'
    f'See docs/stage4_iir_vocabulary.md for the categorization scheme.'
  )


def test_lowerings_rawstring_count_pinned_exact() -> None:
  '''If the count DROPS below the cap, update _MAX_RAWSTRING_CALLS in
  the same commit that lowered it. This test enforces that.'''
  source = _LOWERINGS_PATH.read_text()
  count = source.count('RawString(')

  assert count == _MAX_RAWSTRING_CALLS, (
    f'RawString count is {count}; the cap _MAX_RAWSTRING_CALLS is '
    f'{_MAX_RAWSTRING_CALLS}. If you removed RawString uses, update the '
    f'cap in tests/test_iir_no_raw_string_growth.py to {count} in the same '
    f'commit. This test is intentionally strict — Stage 4 ratchets the '
    f'count down monotonically.'
  )
