'''Verifier infrastructure.

Each dialect ships a verifier callable that runs after every pass on
that dialect's IR shapes. Verifiers catch type mismatches, broken
invariants, malformed IR — anything the pass should not have produced.

The infra is intentionally thin: an error type, plus an aggregator the
PassDriver invokes. Per-dialect verifier logic lives in the dialect's
own module.

See docs/ir_lowering_semantics.md, section 22.
'''

from __future__ import annotations

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
