'''Pass kinds and driver.

A pass is a transformation on the IR. Two flavors:

  Lowering — matches an op of one dialect, produces ops of another
             (or of a target dialect). Each Lowering carries the op
             class it matches, an `apply` callable that performs the
             transformation, and a name for diagnostics.

  Rewrite  — matches an op, produces ops of the *same* dialect.
             Used for internal optimizations like the IIR-sorted-array
             count-as-product or hint-narrowing rules.

The PassDriver runs registered passes against a program tree. Stage 1
implementation is a minimal shell that proves the registry is
consulted; real fixpoint logic lands when the first dialect ships.

See docs/ir_lowering_semantics.md, section 21.
'''

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from srdatalog.ir_core.dialect import Compiler


@dataclass
class Lowering:
  '''A lowering rule from one dialect to another.

  Fields:
    matches — the Op subclass this rule matches (e.g. MirColumnJoin).
    apply   — callable taking (op, context) and returning a list of
              replacement ops in the target dialect.
    name    — short identifier for diagnostics and pass tracing.
  '''

  matches: type
  apply: Callable[[Any, Any], list[Any]]
  name: str = ''


@dataclass
class Rewrite:
  '''A rewrite rule within a single dialect.

  Same shape as Lowering, but conventionally produces ops of the same
  dialect as `matches`. Used for in-dialect optimizations.
  '''

  matches: type
  apply: Callable[[Any, Any], list[Any]]
  name: str = ''


class PassDriver:
  '''Runs registered rewrites and lowerings.

  Stage 1 implementation: a minimal shell that walks the registry but
  doesn't yet apply transformations — the real walker arrives with the
  first dialect that needs it (Stage 3, IIR-sorted-array).

  The driver does not know about specific dialects. New dialects
  participate by being registered; the driver consults the registry.
  '''

  def __init__(self, compiler: Compiler) -> None:
    self._compiler = compiler

  def run(self, prog: Any) -> Any:
    '''Run all registered passes against `prog`. Returns the (possibly
    transformed) program.

    Stage 1: no-op. Returns `prog` unchanged. Validates the registry
    can be walked without error.
    '''
    for _dialect in self._compiler.dialects:
      # Stage 1 placeholder. Real loop:
      #   - run rewrites to fixpoint
      #   - apply lowerings
      #   - verify
      # Lands when dialects ship their pass libraries (Stage 3+).
      pass
    return prog
