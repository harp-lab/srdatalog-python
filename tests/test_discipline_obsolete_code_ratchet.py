'''D19 — obsolete code (backward-compat) ratchet tests.

The redesign accumulates "transitional" code that exists ONLY to keep
old paths working while new ones are being built. Examples:

  - `# DEAD CODE` markers on legacy branches in `lowerings/__init__.py`
    (added by C2-C6 + Phase B as each new typed-Pragma / @lowering
    supersedes a branch but the branch can't be deleted until ALL
    consumers migrate).
  - The legacy `lowerings/__init__.py` monolith itself — 2606 LOC
    today; Phase B per-op migrations are supposed to shrink it.
  - `_BUILTIN_BOOL_SHADOW_PRAGMAS` dual-write registries in `dsl.py`
    (A3 removes them).
  - Deprecated bool fields on `mir.ExecutePipeline` (A3 removes).
  - `_force_legacy_branch()` test fixture (Layer 3 removes once
    `USE_DECLARATIVE` covers every chain-eligible MIR op).

D18 already ratchets one specific transitional pattern
(`object.__setattr__` mutation shims). D19 generalizes the same
"ratchet down, never silently grow" discipline to the OTHER
transitional patterns the redesign has accumulated.

The tests below are intentionally CHEAP — they're text scans + LOC
counts. They cost ~5ms in CI. The point is to make accidental
drift visible at PR time, not to do semantic analysis.

Per `docs/code_discipline.md` D19.
'''

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _REPO_ROOT / 'src' / 'srdatalog'


# -----------------------------------------------------------------------------
# (a) DEAD CODE marker ratchet
# -----------------------------------------------------------------------------
#
# Every `# DEAD CODE` marker in production source MUST carry a removal
# condition. Currently the C2-C6 + B-* PRs use plain `DEAD CODE NOTE (CN)`
# comments naming the superseding phase but no explicit Layer 3 deadline.
# That's fine for now; the cap pins the current count so future markers
# can't accumulate without notice.
#
# Decreasing the cap = a Layer 3 cleanup PR that deletes a marker AND
# its enclosing branch. Increasing the cap = a new PR that introduces a
# new marker; CI requires you to bump the cap explicitly + mention it
# in the PR description.
#
# Snapshot taken post-C6 + B-Filter/ConstantBind/Scan/InsertInto merged:
#   sorted_array/lowerings/__init__.py    8
#   core/passes.py (one informational use) 0  (excluded — framework infra)
#                                        ----
#                                  total  8

_MAX_DEAD_CODE_MARKERS = 8


def _count_dead_code_markers() -> int:
  '''Scan production source for `DEAD CODE` substring. Markers inside
  string literals / docstrings count too — that's intentional, since
  the C-PRs put them in function docstrings.
  '''
  count = 0
  for p in _SRC_ROOT.rglob('*.py'):
    text = p.read_text()
    count += text.count('DEAD CODE')
  return count


def test_dead_code_markers_pinned_exact() -> None:
  '''If a Layer 3 cleanup PR removes a DEAD CODE branch, decrement the
  cap in the same commit. If a new PR adds one, increment + motivate
  in the PR description.

  Strict ratchet — no silent slack in either direction.
  '''
  actual = _count_dead_code_markers()
  assert actual == _MAX_DEAD_CODE_MARKERS, (
    f'`DEAD CODE` marker count: {actual}; cap is {_MAX_DEAD_CODE_MARKERS}. '
    f'If you added a marker, bump the cap and motivate in the PR; if '
    f'you removed one (Layer 3 cleanup), decrement to {actual}.'
  )


# -----------------------------------------------------------------------------
# (b) Legacy monolith LOC ratchet
# -----------------------------------------------------------------------------
#
# `src/srdatalog/ir/dialects/relation/sorted_array/lowerings/__init__.py`
# is the 2500+-LOC imperative `lower_scan_pipeline` monolith Phase B
# is migrating per-op into the `lowerings/lower_mir_<op>.py` files.
# Each B-* migration that deletes a legacy `_lower_<op>` helper SHOULD
# decrement this cap.
#
# Today (post-#49 #52 #54 + CJ-single in flight) the monolith is
# ~2606 LOC. Phase B can grow it slightly when adding new dispatch
# blocks (e.g. USE_DECLARATIVE dispatch wiring); cap should be bumped
# in that case and shrunk in the same PR that deletes the legacy
# helper. Net trend is down.

_MAX_MONOLITH_LOC = 2700  # current ~2606; small headroom for in-flight
_MONOLITH = (
  _SRC_ROOT / 'ir' / 'dialects' / 'relation' / 'sorted_array' / 'lowerings' / '__init__.py'
)


def test_lowerings_monolith_loc_ratchet() -> None:
  '''The legacy `lowerings/__init__.py` monolith MUST NOT grow
  unboundedly. Each B-* PR that deletes a `_lower_<op>` helper should
  decrement the cap in the same commit. Phase B "done" = cap reaches
  the import + scaffold floor (~200 LOC estimated).
  '''
  actual = len(_MONOLITH.read_text().splitlines())
  assert actual <= _MAX_MONOLITH_LOC, (
    f'Legacy monolith grew to {actual} LOC (cap {_MAX_MONOLITH_LOC}). '
    f'Either bump the cap in the same commit (and motivate in the PR), '
    f'or shrink the monolith back below {_MAX_MONOLITH_LOC}.'
  )


# -----------------------------------------------------------------------------
# (c) USE_DECLARATIVE completeness gate
# -----------------------------------------------------------------------------
#
# When `USE_DECLARATIVE` covers every chain-eligible MIR op, Phase B
# is done and Layer 3 cleanup can delete the legacy fall-through
# (`raise ValueError('unsupported inner op: ...')`) + the
# `_force_legacy_branch` test fixture.
#
# Today this gate just MONITORS coverage; it doesn't fail on
# incomplete coverage (Phase B is in flight). When the coverage hits
# 100%, the test starts asserting cleanup conditions.

# Chain-eligible = MIR ops that legitimately appear inside an
# ExecutePipeline.pipeline (i.e. would route through `_lower_inner_chain`).
# Excludes structural / fixpoint-maintenance / wrap ops.
_CHAIN_ELIGIBLE_OP_NAMES = frozenset(
  {
    'ColumnSource',  # root-position only; covered by Scan slot in spec
    'Scan',
    'ColumnJoin',
    'CartesianJoin',
    'Filter',
    'ConstantBind',
    'Negation',
    'Aggregate',
    'InsertInto',
    'CreateFlatView',  # split-rule path
    'InnerPipeline',  # nested join
    'ProbeJoin',  # binary-join mode
    'GatherColumn',  # binary-join mode
  }
)


def _use_declarative_op_names() -> set[str]:
  '''Read `USE_DECLARATIVE` from sorted_array via import (lightweight).'''
  from srdatalog.ir.dialects.relation.sorted_array import USE_DECLARATIVE

  return {cls.__name__ for cls in USE_DECLARATIVE}


def test_use_declarative_chain_coverage_monitor() -> None:
  '''Informational: how many chain-eligible MIR ops are migrated.
  Asserts non-strict monotonic coverage (the count can only grow).

  When coverage == 100%, the SECOND assertion below activates and
  guards Layer 3 cleanup.
  '''
  covered = _use_declarative_op_names() & _CHAIN_ELIGIBLE_OP_NAMES
  total = len(_CHAIN_ELIGIBLE_OP_NAMES)

  # Floor — protects against accidental regression (someone removes
  # an op from USE_DECLARATIVE). Increment in the same commit when a
  # new B-* PR adds a migration.
  _MIN_COVERAGE = 4  # post-#49 #52 #54 (Filter, ConstantBind, Scan, InsertInto)
  assert len(covered) >= _MIN_COVERAGE, (
    f'USE_DECLARATIVE coverage dropped to {len(covered)} / {total} '
    f'({sorted(covered)}); floor is {_MIN_COVERAGE}. '
    f'Did a B-* migration accidentally regress?'
  )


def test_use_declarative_completeness_gate_layer3() -> None:
  '''Layer 3 cleanup gate: when USE_DECLARATIVE covers all chain-eligible
  ops, the legacy fall-through `raise ValueError('unsupported inner op')`
  in `lowerings/__init__.py` becomes unreachable and MUST be deleted.

  Today this is a no-op (Phase B in flight); it activates when
  coverage == 100%.
  '''
  covered = _use_declarative_op_names() & _CHAIN_ELIGIBLE_OP_NAMES
  if covered == _CHAIN_ELIGIBLE_OP_NAMES:
    # Phase B is done — legacy fall-through must be gone.
    monolith_text = _MONOLITH.read_text()
    legacy_marker = "raise ValueError(f'unsupported inner op:"
    assert legacy_marker not in monolith_text, (
      'Phase B is complete (USE_DECLARATIVE covers every chain-eligible '
      'MIR op) but the legacy fall-through is still in '
      'lowerings/__init__.py. Layer 3 cleanup PR must delete:\n'
      '  - the `raise ValueError(...unsupported inner op...)` line\n'
      '  - the `_lower_<op>` helpers for every migrated op\n'
      '  - the `_force_legacy_branch` test fixture in test_lower_mir_*\n'
      '  - the `USE_DECLARATIVE` frozenset itself (now redundant)\n'
      'See docs/code_discipline.md D19 for the cleanup checklist.'
    )
