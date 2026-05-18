'''`MirPragmaPass` — typed pragma materialization for MIR.

Walks every `ExecutePipeline` in an MIR program and, for each, applies
the registered `@pragma_handler` callbacks in Kahn-topo-sorted order
to consume each present `Pragma` instance into a wrap-op insertion.
After the pass, every EP's `pragmas` field MUST be empty —
`UnconsumedPragmaError` fires otherwise.

Spec: docs/phase_c_pragma_materialization.md §3 (algorithm) and
docs/pragma_as_typed_object.md §5 (type-based dispatch).

Phase C1 ships this pass with **zero registered handlers**, so the
walk is a structural identity for every EP currently produced by
`HirToMirLowering`. C2-C6 each migrate one of the named bool fields
(`dedup_hash`, `block_group`, `work_stealing`, `tiled_cartesian`,
`count`, `fanout`) into a typed `Pragma` subclass + a registered
handler; the discipline test
`test_pragmas_empty_after_materialization` (this file's test
companion) gates the migration.

Why a separate module from `mir/passes.py`: `mir/passes.py` is the
home for *optimization* passes over MIR (clause reorder, prefix
reorder, concurrent-write marking). Pragma materialization is a
structurally different transformation (it consumes typed
declarations and inserts wrap ops); per the spec it deserves its own
file so future readers can find it under the obvious name.

Per discipline D10 / parallel rule for `LowerCtx`, the per-pass
context object (`PragmaCtx`) is minimal — it carries only the active
`Compiler`. Handlers that need richer state get amendments to
`PragmaCtx`, not free-form mutable state.
'''

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from srdatalog.ir.core.ops import Op
from srdatalog.ir.core.passes import RewritePass
from srdatalog.ir.core.pragma import (
  Pragma,
  PragmaCtx,
  PragmaOrderingError,
  PragmaRegistration,
  UnconsumedPragmaError,
  UnregisteredPragmaError,
  get_pragma_registrations,
)
from srdatalog.ir.core.strategy import bottom_up
from srdatalog.ir.mir.types import ExecutePipeline

# -----------------------------------------------------------------------------
# Topological sort over Pragma classes
# -----------------------------------------------------------------------------


def _topo_sort_pragma_classes(
  present: list[type[Pragma]],
  registry: dict[type[Pragma], PragmaRegistration],
) -> list[type[Pragma]]:
  '''Kahn-topo-sort the given pragma classes by registered `before` /
  `after` constraints.

  `present` is the (deduplicated, registration-order) list of pragma
  classes actually present on the EP being processed. `registry`
  maps each present class to its registration. Edges:

    - A `before=(B,)` declaration on A's registration adds A -> B
      (A must run before B).
    - An `after=(A,)` declaration on B's registration adds A -> B
      (A must run before B), symmetric encoding.

  Edges that reference a class not in `present` are dropped (the
  ordering only matters between classes that are actually firing on
  this EP). Cycles raise `PragmaOrderingError` naming the cycle.

  Tie-break is deterministic registration order from `present` so
  the output is stable across Python hash-seed changes (Kahn's
  algorithm with a stable priority on the ready set).
  '''
  present_set = set(present)
  # Adjacency: edges[u] = set of v such that u must run before v.
  edges: dict[type[Pragma], set[type[Pragma]]] = {cls: set() for cls in present}
  in_degree: dict[type[Pragma], int] = {cls: 0 for cls in present}

  for cls in present:
    reg = registry[cls]
    for nxt in reg.before:
      if nxt in present_set and nxt not in edges[cls]:
        edges[cls].add(nxt)
        in_degree[nxt] += 1
    for prv in reg.after:
      if prv in present_set and cls not in edges[prv]:
        edges[prv].add(cls)
        in_degree[cls] += 1

  # Kahn: pick zero-in-degree nodes in registration order.
  order: list[type[Pragma]] = []
  remaining = list(present)
  while remaining:
    ready = [cls for cls in remaining if in_degree[cls] == 0]
    if not ready:
      raise PragmaOrderingError(
        f'cycle in @pragma_handler before/after constraints among {[c.__name__ for c in remaining]}'
      )
    pick = ready[0]
    order.append(pick)
    remaining.remove(pick)
    for v in edges[pick]:
      in_degree[v] -= 1

  return order


# -----------------------------------------------------------------------------
# MirPragmaPass
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class MirPragmaPass(RewritePass):
  '''One-shot RewritePass: materialize every typed `Pragma` instance.

  Walks the MIR program; for every `ExecutePipeline`, runs the
  registered `@pragma_handler` callbacks (Kahn-topo-sorted by
  `before` / `after`) in turn until `ep.pragmas == ()`.

  C1 no-op behavior: with zero registered handlers, the walk visits
  every EP, the per-EP topo-sort is empty, and the post-flight
  invariant (`ep.pragmas == ()`) holds trivially since every EP
  currently produced by `HirToMirLowering` has an empty `pragmas`
  tuple. The pass is structurally identity until C2 lands.

  Algorithm (per spec §3):

    1. Snapshot the global `@pragma_handler` registry.
    2. Walk the program bottom-up; for each `ExecutePipeline`:
       a. Identify the present pragma classes (deduplicated).
       b. Pre-flight: every present class must have a registration,
          else `UnregisteredPragmaError` (with did-you-mean hint).
       c. Topo-sort the present classes.
       d. For each class in order: invoke the handler. The handler
          returns a transformed EP (with the consumed pragma stripped
          from `pragmas`).
       e. Post-flight: `ep.pragmas == ()`, else `UnconsumedPragmaError`.

  Spec: docs/phase_c_pragma_materialization.md §3,
  docs/pragma_as_typed_object.md §5.
  '''

  name: str = 'mir_pragma_materialization'
  consumes: tuple[str, ...] = ('mir',)
  produces: tuple[str, ...] = ('mir',)
  dialect_name: str = 'mir'
  until_fixpoint: bool = False

  def apply(self, prog: Any, compiler: Any) -> Any:
    registrations = get_pragma_registrations()
    # Restrict the registry to handlers targeting `ExecutePipeline`.
    # Other ops in MIR don't carry pragmas today; if a future op
    # gains a `pragmas` field, the handler's `on=` will pick it up
    # and this filter expands accordingly.
    registry: dict[type[Pragma], PragmaRegistration] = {}
    for reg in registrations:
      if reg.on is not ExecutePipeline:
        continue
      # Last writer wins for duplicate registrations on the same
      # Pragma class — symmetric with how the rest of the codebase
      # handles multi-decoration; in practice C2-C6 each register
      # exactly one handler per Pragma type.
      registry[reg.pragma_cls] = reg

    ctx = PragmaCtx(compiler=compiler)

    def _materialize(op: Op) -> Op | None:
      if not isinstance(op, ExecutePipeline):
        return None
      if not op.pragmas:
        return None
      return _materialize_one_ep(op, registry, ctx)

    return bottom_up(_materialize)(prog)


def _materialize_one_ep(
  ep: ExecutePipeline,
  registry: dict[type[Pragma], PragmaRegistration],
  ctx: PragmaCtx,
) -> ExecutePipeline:
  '''Materialize all pragmas on a single ExecutePipeline.

  Per-EP algorithm: identify present pragma classes, validate every
  present class has a handler (`UnregisteredPragmaError` otherwise),
  topo-sort by `before` / `after`, invoke each handler in order,
  assert post-flight `ep.pragmas == ()`
  (`UnconsumedPragmaError` otherwise).

  Returns the transformed EP. The handler for each pragma is
  responsible for stripping its own `Pragma` instance from
  `ep.pragmas` (otherwise the post-flight assertion fires; the
  handler is buggy).
  '''
  present = _present_pragma_classes(ep.pragmas)

  # Pre-flight: every present pragma must be registered.
  for cls in present:
    if cls not in registry:
      known = sorted(c.__name__ for c in registry)
      raise UnregisteredPragmaError(
        f'{cls.__name__} appears on ExecutePipeline {ep.rule_name!r} '
        f'but no @pragma_handler claims its type. '
        f'Did you mean one of: {known}? '
        f'(import the module that registers the handler, or add '
        f'`@pragma_handler({cls.__name__}, on=ExecutePipeline)` to '
        f'a loaded module.)'
      )

  ordered = _topo_sort_pragma_classes(present, registry)

  current = ep
  for cls in ordered:
    reg = registry[cls]
    instance = _first_instance(current.pragmas, cls)
    if instance is None:
      # A previous handler in this EP already consumed it (or it
      # never appeared on this op). Skip.
      continue
    result = reg.fn(current, instance, ctx)
    if result is None:
      # Spec §2: handler may return None to skip. The pragma stays
      # on the EP; the post-flight check below will then fire as
      # `UnconsumedPragmaError`.
      continue
    current = result

  # Post-flight: every pragma must have been consumed.
  if current.pragmas:
    surviving = current.pragmas[0]
    handler_names = sorted(c.__name__ for c in registry)
    raise UnconsumedPragmaError(
      f'pragma {surviving!r} on ExecutePipeline {current.rule_name!r} '
      f'survived MirPragmaPass; the registered handler for '
      f'{type(surviving).__name__} returned an op without removing '
      f'this instance from `pragmas`. Registered handlers: '
      f'{handler_names}.'
    )

  return current


def _present_pragma_classes(pragmas: Iterable[Any]) -> list[type[Pragma]]:
  '''Distinct pragma TYPES present on `pragmas`, in first-seen order.

  Filters to actual `Pragma` instances; non-Pragma entries (the
  legacy name-keyed `(str, Any)` shape that still types
  `ExecutePipeline.pragmas` until A3) are skipped. C1 behavior: the
  field is empty in practice, so this returns `[]`.
  '''
  seen: dict[type[Pragma], None] = {}
  for p in pragmas:
    if isinstance(p, Pragma) and type(p) not in seen:
      seen[type(p)] = None
  return list(seen)


def _first_instance(
  pragmas: Iterable[Any],
  cls: type[Pragma],
) -> Pragma | None:
  '''First `cls` instance in `pragmas`, or None.

  Mirrors `core.pragma.get_pragma` but operates on the iterable
  directly (no Op wrapper required) so we can call it on the
  in-progress `current.pragmas` after each handler.
  '''
  for p in pragmas:
    if isinstance(p, cls):
      return p
  return None


__all__ = [
  'MirPragmaPass',
]
