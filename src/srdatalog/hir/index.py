'''HIR Pass 5: Index Selection.

Populates per-stratum `required_indices` / `canonical_index` and program-level
`global_index_map`. Required for downstream MIR lowering, which embeds the
canonical index columns into RebuildIndex / ComputeDelta / MergeIndex etc.

Mirrors src/srdatalog/hir/index_selection.nim.

Algorithm:
  1. First pass: collect every index used by any variant in any stratum
     into `hir.global_index_map`.
  2. Second pass: for each stratum, per SCC member:
     - Recursive: DELTA-version indices first, FULL-version second,
       global fallback, identity fallback.
     - Non-recursive: local patterns, global fallback, identity fallback.
  3. Canonical index: prefer a full-arity index that is also accessed as
     FULL in the stratum (so the maintained FULL-version index is actually
     read; avoids building a dead index).

SCC member iteration is sorted for reproducibility (Python set order is
hash-random); Nim relies on its deterministic string hash, but these
fields are internal to downstream passes and not emitted, so the order
only has to be stable, not byte-matched against Nim.
'''
from __future__ import annotations

from srdatalog.hir.types import (
  HirProgram,
  HirRuleVariant,
  RelationDecl,
  Version,
)
from srdatalog.hir.pass_ import PassInfo, PassLevel, Dialect


def get_arity(rel_name: str, decls: list[RelationDecl]) -> int:
  for d in decls:
    if d.rel_name == rel_name:
      return len(d.types)
  return 0


def default_index(rel_name: str, decls: list[RelationDecl]) -> list[int]:
  return list(range(get_arity(rel_name, decls)))


def complete_index(idx: list[int], arity: int) -> list[int]:
  '''Pad a partial index to full arity by appending missing columns in column order.'''
  result = list(idx)
  for col in range(arity):
    if col not in result:
      result.append(col)
  return result


def canonical_index(
  rel_name: str,
  indices: list[list[int]],
  decls: list[RelationDecl],
  full_indices: dict[str, set[tuple[int, ...]]] | None = None,
) -> list[int]:
  '''Pick a full-arity canonical index. Prefer one whose FULL-version is
  actually read by joins (so the maintained FULL index is actively used).
  '''
  arity = get_arity(rel_name, decls)
  if full_indices and rel_name in full_indices:
    for idx in indices:
      if len(idx) == arity and tuple(idx) in full_indices[rel_name]:
        return list(idx)
  for idx in indices:
    if len(idx) == arity:
      return list(idx)
  return default_index(rel_name, decls)


def _collect_indices_from_variants(
  variants: list[HirRuleVariant], version: Version | None = None
) -> dict[str, set[tuple[int, ...]]]:
  '''Gather {rel_name: {tuple(index_cols), ...}} from access and negation
  patterns, optionally filtered to a specific Version.
  '''
  out: dict[str, set[tuple[int, ...]]] = {}
  for v in variants:
    for pat in v.access_patterns:
      if version is None or pat.version is version:
        out.setdefault(pat.rel_name, set()).add(tuple(pat.index_cols))
    for pat in v.negation_patterns:
      if version is None or pat.version is version:
        out.setdefault(pat.rel_name, set()).add(tuple(pat.index_cols))
  return out


def _append_unique_indices(
  dest: list[list[int]], source_tuples: set[tuple[int, ...]], arity: int
) -> None:
  '''Append completed indices to `dest` in sorted order, skipping duplicates.'''
  for t in sorted(source_tuples):
    cidx = complete_index(list(t), arity)
    if cidx not in dest:
      dest.append(cidx)


def select_indices(hir: HirProgram) -> HirProgram:
  '''Pass 5 entry. Mutates and returns the HirProgram.'''
  decls = hir.relation_decls

  # ----- First pass: global_index_map over all strata.
  for stratum in hir.strata:
    all_variants = stratum.base_variants + stratum.recursive_variants
    all_idx = _collect_indices_from_variants(all_variants, version=None)
    for rel_name in sorted(all_idx.keys()):
      if rel_name not in hir.global_index_map:
        hir.global_index_map[rel_name] = []
      for t in sorted(all_idx[rel_name]):
        idx_list = list(t)
        if idx_list not in hir.global_index_map[rel_name]:
          hir.global_index_map[rel_name].append(idx_list)

  # ----- Second pass: per-stratum required + canonical.
  for stratum in hir.strata:
    if stratum.is_recursive:
      delta_idx = _collect_indices_from_variants(stratum.recursive_variants, Version.DELTA)
      full_idx = _collect_indices_from_variants(stratum.recursive_variants, Version.FULL)
    else:
      delta_idx = {}
      full_idx = {}
      all_idx = _collect_indices_from_variants(stratum.base_variants, version=None)

    for rel_name in sorted(stratum.scc_members):
      arity = get_arity(rel_name, decls)
      indices: list[list[int]] = []

      if stratum.is_recursive:
        if rel_name in delta_idx:
          _append_unique_indices(indices, delta_idx[rel_name], arity)
        if rel_name in full_idx:
          _append_unique_indices(indices, full_idx[rel_name], arity)
      else:
        if rel_name in all_idx:
          _append_unique_indices(indices, all_idx[rel_name], arity)

      # Fallback 1: global index map (indices used by OTHER strata).
      if not indices and rel_name in hir.global_index_map:
        for idx in hir.global_index_map[rel_name]:
          cidx = complete_index(idx, arity)
          if cidx not in indices:
            indices.append(cidx)

      # Fallback 2: identity index.
      if not indices:
        indices.append(default_index(rel_name, decls))

      stratum.required_indices[rel_name] = indices
      stratum.canonical_index[rel_name] = canonical_index(
        rel_name, indices, decls,
        full_indices=full_idx if stratum.is_recursive else None,
      )

  return hir


class IndexSelectionPass:
  info = PassInfo(
    name="IndexSelection",
    level=PassLevel.HIR_TRANSFORM,
    order=300,
    source_dialect=Dialect.HIR,
    target_dialect=Dialect.HIR,
  )

  def run(self, hir: HirProgram) -> HirProgram:
    return select_indices(hir)
