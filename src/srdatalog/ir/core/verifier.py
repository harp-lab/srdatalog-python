'''Verifier infrastructure.

Each dialect ships a verifier callable that runs after every pass on
that dialect's IR shapes. Verifiers catch type mismatches, broken
invariants, malformed IR — anything the pass should not have produced.

The infra is intentionally thin: an error type, plus an aggregator the
PassDriver invokes. Per-dialect verifier logic lives in the dialect's
own module.

This module also hosts `verify_renderability` (R3 — the post-IIR closure
check). For every op type reachable in the post-fixpoint IIR tree,
EITHER a `@register_render(op_type, target=T)` is registered OR a
`@rewrite(dialect, op_type)` is registered. The check raises
`UnrenderableOpError` on the first op type that satisfies neither
condition. This is "topology check #4" in docs/concept_glossary.md
section 13.3 — it polices the SOFT boundary within an IR stage
(multi-dialect IIR tree closure).

See docs/ir_lowering_semantics.md, section 22 and
docs/concept_glossary.md section 13.6 (the closure invariant).
'''

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any


@dataclass
class VerificationError:
  '''An IR well-formedness violation.

  Fields:
    op      — the offending op instance.
    message — human-readable description of the violation.
  '''

  op: Any
  message: str

  def __str__(self) -> str:
    return f'VerificationError on {type(self.op).__name__}: {self.message}'


# -----------------------------------------------------------------------------
# R3 closure check — verify_renderability
# -----------------------------------------------------------------------------


class UnrenderableOpError(Exception):
  '''An IIR op type has neither a `@register_render(op_type, target=T)`
  handler nor a `@rewrite(dialect, op_type)` registered.

  Raised by `verify_renderability` post-fixpoint, before the target
  renderer runs. This is the loud failure that makes adding a new IIR
  dialect / plugin safe: if the plugin forgot to ship either a renderer
  for the target or a decomposing rewrite that maps its ops to ones
  the target understands, this error fires with a precise op-type name
  and target — long before the renderer's `KeyError` would surface.

  Per docs/concept_glossary.md section 13.6 (the closure invariant) and
  docs/code_discipline.md R3.
  '''

  def __init__(self, op_type: type, target: str) -> None:
    self.op_type = op_type
    self.target = target
    module = getattr(op_type, '__module__', '<unknown>')
    super().__init__(
      f'no @register_render({op_type.__name__}, target={target!r}) and no '
      f'@rewrite registration found for op type {op_type.__name__!r} '
      f'(module={module!r}). Either ship a renderer in '
      f'codegen/{target}/render/<dialect>.py or register a @rewrite that '
      f'decomposes this op into ones the target can render.'
    )


def _walk_op_types(tree: Any) -> set[type]:
  '''Collect every `type(op)` reachable from `tree` by walking
  dataclass fields. Mirrors `core.strategy._map_children`'s child-
  discovery rule: an `Op` field is a child, and `list` / `tuple`
  fields contribute every Op they contain. Non-Op fields are ignored.

  Returns a set of op type objects (Op subclasses).
  '''
  # Local import to avoid the framework-init cycle between
  # verifier.py and ops.py at module-load time.
  from srdatalog.ir.core.ops import Op

  seen: set[type] = set()
  stack: list[Any] = [tree]
  while stack:
    node = stack.pop()
    if isinstance(node, Op):
      seen.add(type(node))
      # Op subclasses are dataclasses (D2); discover children by
      # walking their fields. Per the strategy combinator's
      # convention, an Op-valued field is a child; list/tuple fields
      # may contain Ops.
      for f in dataclasses.fields(node):
        stack.append(getattr(node, f.name))
    elif isinstance(node, (list, tuple)):
      stack.extend(node)
    # Other field shapes (str, int, dict-of-primitives, ...) are ignored.
  return seen


def _renderable_op_types(target: str) -> set[type]:
  '''Return the set of op classes that have a `@register_render` for
  `target` in either statement OR expression mode. Currently only
  target='cuda' is wired into the codebase; other targets fall back
  to the empty set.

  Per docs/stage3a_execution_plan.md S3A.8, the per-target renderer
  registry will eventually move onto a per-`Codegen` instance. Until
  then, the CUDA registry is the module-globals
  `_STMT_HANDLERS` + `_EXPR_HANDLERS` in `codegen/cuda/render`.
  '''
  if target != 'cuda':
    return set()
  # Local import to avoid pulling codegen at framework-init time.
  from srdatalog.ir.codegen.cuda.render import _EXPR_HANDLERS, _STMT_HANDLERS

  return set(_STMT_HANDLERS.keys()) | set(_EXPR_HANDLERS.keys())


def _rewritable_op_types(compiler: Any) -> set[type]:
  '''Return op classes that have a `@rewrite(dialect, op_type)`
  registered on any of the compiler's dialects. Used as the closure
  escape hatch: a COMPOUND op without a target renderer still counts
  as "renderable" if a rewrite decomposes it.

  When `compiler` is None, no per-Compiler rewrites are visible. The
  caller must also pass a compiler to consult plugin-registered
  rewrites; without one, only the global render registry gates closure.
  '''
  if compiler is None:
    return set()
  out: set[type] = set()
  for d in compiler.dialects:
    for r in d.rewrites:
      out.add(r.matches)
  return out


def verify_renderability(
  tree: Any,
  *,
  target: str = 'cuda',
  compiler: Any = None,
) -> None:
  '''Closure check: every op type in `tree` must EITHER have a
  registered `@register_render(op_type, target=target)` OR a
  registered `@rewrite(dialect, op_type)` (in which case the rewrite
  is presumed to decompose the op into ones that close under the
  same condition — the recursive guarantee R3 makes).

  Walks `tree` once, collecting reachable op types, then verifies
  each is in the union of the two registries. Raises
  `UnrenderableOpError` on the first unrenderable op type, naming
  the op + target so the caller can either:

    (a) ship a renderer at `codegen/<target>/render/<dialect>.py`, or
    (b) ship a `@rewrite` that decomposes the op into ones the target
        already renders.

  Args:
    tree     — the post-fixpoint IIR root (any `Op` subclass).
    target   — codegen target name. Defaults to 'cuda' (the only
               wired-in target today). Other targets return the empty
               renderer set, which means EVERY op must have a
               rewrite — useful for testing that a non-CUDA target
               flow degrades gracefully.
    compiler — optional `Compiler` whose registered dialects'
               rewrites are consulted as the escape hatch. When None,
               only the renderer registry is consulted (production
               pipelines pass the active compiler so plugin-shipped
               rewrites count).

  Per docs/concept_glossary.md section 13.6 and docs/code_discipline.md
  R3 — the closure invariant for multi-dialect IIR.
  '''
  reachable = _walk_op_types(tree)
  renderable = _renderable_op_types(target)
  rewritable = _rewritable_op_types(compiler)
  closed = renderable | rewritable

  for op_type in reachable:
    if op_type not in closed:
      raise UnrenderableOpError(op_type, target)


__all__ = [
  'UnrenderableOpError',
  'VerificationError',
  'verify_renderability',
]
