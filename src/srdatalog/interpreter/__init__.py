'''Reference interpreters for IR semantic functions.

Direct executable implementations of the denotational semantics from
docs/ir_lowering_semantics.md. Used as ground truth for
property-based tests of lowerings and rewrites.

Stage 2 plan:
  mir.py  — implements [[ . ]]_M from section 6.

Later stages add per-dialect interpreters (e.g. iir_sorted_array.py)
to validate dialect-specific lowerings against the same MIR-level
ground truth.
'''
