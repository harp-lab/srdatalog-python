'''Data-structure dialects.

Each subpackage is one dialect describing how a relation is stored
and accessed (sorted-array, LSM, union-find, bitmap, …). New
data-structure dialects are added as siblings — no changes to other
categories required (Property P1).

Currently registered:
  - sorted_array: contiguous sorted column arrays, NodeHandle navigation.

Planned:
  - lsm: K-level Log-Structured Merge (audit gap G3 + spec §13).
  - uf: union-find for equivalence-class relations (spec §14).
  - bitmap: dense small-domain relations (spec §15).
'''
