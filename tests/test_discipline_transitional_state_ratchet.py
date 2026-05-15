'''D18 — transitional mutable state ratchet test.

Per `docs/code_discipline.md` §2 D18 + `docs/phase_zero_prerequisites.md`
§2.1: any `object.__setattr__(...)` call on a frozen `Op` subclass is
transitional tech-debt that MUST be removed in a named follow-up
phase. The cap below pins the current count; CI fails if the count
goes UP.

This is the V1 scaffold — covers sub-category (a) only:

  (a) `object.__setattr__` shims on frozen Op subclasses (THIS TEST)
  (b) `# DEPRECATED:` fields on dataclasses pending removal (FUTURE)
  (c) module-global mutable registries with per-Compiler target (FUTURE)

(b) and (c) extend this test file when those patterns are established.

Same ratchet shape as D5 (`test_iir_no_raw_string_growth`, whole-tree
scan) and D12 (`USE_DECLARATIVE` monotonic-add). The cap stays at or
below the current count; A2/A3 reduces it; Layer 3 cleanup hits zero.

A1's inventory baseline (from `docs/phase_a_mir_onto_op.md` §8):
  - `ir/codegen/cuda/envelope.py` — `_assign_handle_positions_rec`
  - `ir/codegen/cuda/pipeline_utils.py` — `_assign_handle_positions_rec` twin
  - `ir/codegen/cuda/orchestrator.py` — `_gen_parallel_group` concurrent_write
  - `ir/mir/passes.py` — reorder / source-spec / balanced-scan passes
'''

from __future__ import annotations

from pathlib import Path

# Whole IR + codegen tree. The shim pattern is only relevant inside
# `src/srdatalog/ir/` (frozen-Op land); other packages don't have a
# frozen-Op contract, so they don't need to be ratcheted here.
_SRC_ROOTS = (Path(__file__).resolve().parent.parent / 'src' / 'srdatalog' / 'ir',)

# Files explicitly excluded from the ratchet. Anything listed here is
# permanent framework infra (NOT a transition shim) and must carry an
# inline rationale in the source file plus an entry in D18 of
# `docs/code_discipline.md`.
#
# Currently empty: every `object.__setattr__(` call site under
# `src/srdatalog/ir/` at the time this test ships is an A1-era
# transition shim. `core/pragma.py` does not use the shim pattern
# (the typed-Pragma registry is built by class-body decorators, not
# instance mutation).
_EXCLUDED: frozenset[Path] = frozenset()


def _all_shim_sites() -> dict[Path, int]:
  '''Return ``{path: count}`` for every src file under `_SRC_ROOTS`
  that contains `object.__setattr__(` calls (excluding `_EXCLUDED`).
  '''
  out: dict[Path, int] = {}
  for root in _SRC_ROOTS:
    for p in root.rglob('*.py'):
      if p in _EXCLUDED:
        continue
      text = p.read_text()
      n = text.count('object.__setattr__(')
      if n > 0:
        out[p] = n
  return out


# RATCHET: only DECREASE this. Increases require explicit owner
# sign-off + amendment of D18 in `docs/code_discipline.md`.
#
# Snapshot post-A2.2 (orchestrator concurrent_write shim removed via
# the new `apply_concurrent_write_marking` MIR pass; rebased on top
# of A2.1 which already removed the 6 envelope + pipeline_utils shims):
#   ir/mir/passes.py                      10
#   ir/codegen/cuda/orchestrator.py        0  (was 1)
#                                       ----
#                              total      10
#
# A2.3 removes the mir/passes.py reorder shims (10 → 0).
_MAX_TRANSITIONAL_SHIMS = 10


def test_no_transitional_shim_growth() -> None:
  sites = _all_shim_sites()
  total = sum(sites.values())
  if total > _MAX_TRANSITIONAL_SHIMS:
    repo_root = _SRC_ROOTS[0].parent.parent.parent
    breakdown = '\n'.join(f'  {p.relative_to(repo_root)}: {n}' for p, n in sorted(sites.items()))
    raise AssertionError(
      f'object.__setattr__(...) shim count: {total}, cap is '
      f'{_MAX_TRANSITIONAL_SHIMS}.\n'
      f'Breakdown:\n{breakdown}\n\n'
      'Per D18 (`docs/code_discipline.md` §2), any new '
      '`object.__setattr__`\n'
      'on a frozen Op subclass is transitional tech-debt. Either:\n'
      '  (a) Remove an existing shim (preferred) and update the cap.\n'
      '  (b) Add a `# TODO(phase-X): <action>` comment and motivate\n'
      '      the new shim in the PR description.\n'
      '  (c) If the new use is genuinely permanent (framework infra,\n'
      '      not a transition), exclude the file in `_EXCLUDED` + amend\n'
      '      D18 in `docs/code_discipline.md`.\n'
      'See `docs/phase_zero_prerequisites.md` §2.1 for the contract.'
    )


def test_shim_count_pinned_exact() -> None:
  '''If shim count drops below the cap, update `_MAX_TRANSITIONAL_SHIMS`
  in the same commit. Strict ratchet — no silent slack.'''
  total = sum(_all_shim_sites().values())
  assert total == _MAX_TRANSITIONAL_SHIMS, (
    f'object.__setattr__ shim count is {total}; cap is '
    f'{_MAX_TRANSITIONAL_SHIMS}. If you removed a shim, update the '
    f'cap to {total} in the same commit.'
  )
