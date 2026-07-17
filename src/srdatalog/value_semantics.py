'''Semantic declarations for functional, lattice-valued relations.

These declarations belong above MIR: they say when two tuples denote the
same logical fact and how competing values combine.  They deliberately do
not choose a physical index (hash table, sorted array, or otherwise).
'''

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import Enum


class LatticeJoin(Enum):
  '''Join operation used to combine values for the same logical key.'''

  INTERVAL_INTERSECTION = "interval-intersection"
  MAX_LOWER_SELECT = "max-lower-select"


class ValueEncoding(Enum):
  '''Scalar encoding used by value columns in the physical tuple.'''

  FLOAT32_BITS = "float32-bits"
  UINT32_WORDS = "uint32-words"


@dataclass(frozen=True)
class LatticeValueSpec:
  '''A functional dependency ``key_columns -> value_columns``.

  ``join`` defines the monotone information-order update.  For interval
  intersection, ``[l1, u1] join [l2, u2]`` is
  ``[max(l1, l2), min(u1, u2)]``.  A changed joined value, rather than every
  candidate tuple, is the semantic semi-naive delta.
  '''

  key_columns: tuple[int, ...]
  value_columns: tuple[int, ...]
  join: LatticeJoin
  encoding: ValueEncoding

  def validate(self, arity: int) -> None:
    if not self.key_columns:
      raise ValueError("lattice value key_columns must not be empty")
    if not self.value_columns:
      raise ValueError("lattice value value_columns must not be empty")
    columns = self.key_columns + self.value_columns
    if len(set(columns)) != len(columns):
      raise ValueError("lattice key and value columns must be distinct")
    if set(columns) != set(range(arity)):
      raise ValueError(
        "lattice key and value columns must partition relation columns "
        f"0..{arity - 1}; got {columns}"
      )
    if self.join is LatticeJoin.INTERVAL_INTERSECTION and len(self.value_columns) != 2:
      raise ValueError("interval-intersection requires lower and upper value columns")
    if self.join is LatticeJoin.MAX_LOWER_SELECT and len(self.value_columns) != 3:
      raise ValueError("max-lower-select requires rank, lower, and upper value columns")


def interval_lattice(
  *,
  key_columns: tuple[int, ...],
  lower_column: int,
  upper_column: int,
  encoding: ValueEncoding = ValueEncoding.FLOAT32_BITS,
) -> LatticeValueSpec:
  '''Declare an interval-valued functional relation.'''
  return LatticeValueSpec(
    key_columns=tuple(key_columns),
    value_columns=(lower_column, upper_column),
    join=LatticeJoin.INTERVAL_INTERSECTION,
    encoding=encoding,
  )


def max_lower_lattice(
  *,
  key_columns: tuple[int, ...],
  rank_column: int,
  lower_column: int,
  upper_column: int,
  encoding: ValueEncoding = ValueEncoding.UINT32_WORDS,
) -> LatticeValueSpec:
  '''Declare materialized state for a grouped maximum-lower witness aggregate.

  The greatest lower bound wins; minimum stable rank resolves an exact lower
  tie.  Rank is a witness identity/tie-break key, not an evidence score.  For
  the selection to be a deterministic join, ``key + rank`` must identify one
  interval payload.
  '''
  return LatticeValueSpec(
    key_columns=tuple(key_columns),
    value_columns=(rank_column, lower_column, upper_column),
    join=LatticeJoin.MAX_LOWER_SELECT,
    encoding=encoding,
  )


def float32_to_u32(value: float) -> int:
  '''Bit-cast a probability bound to an unsigned integer of the same size.'''
  if not 0.0 <= value <= 1.0:
    raise ValueError(f"interval bound must be in [0,1], got {value}")
  return struct.unpack("<I", struct.pack("<f", value))[0]


def u32_to_float32(bits: int) -> float:
  '''Invert :func:`float32_to_u32`.'''
  if not 0 <= bits <= 0xFFFFFFFF:
    raise ValueError(f"float32 bit pattern must fit uint32, got {bits}")
  return struct.unpack("<f", struct.pack("<I", bits))[0]
