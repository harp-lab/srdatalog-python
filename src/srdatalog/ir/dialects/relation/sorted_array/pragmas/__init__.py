'''Sorted-array dialect pragmas.

Per `docs/phase_c_pragma_materialization.md` §4.3, pragmas that
specialize sorted-array emission (currently `dedup_hash`; the
similarly tightly-coupled `tiled_cartesian` lands in C5) live under
this dialect's `pragmas/` subpackage. Each module defines:

  - one `Pragma` subclass (typed compile-time object, per
    `docs/pragma_as_typed_object.md` §2),
  - one `@pragma_handler(PragmaCls, on=mir.ExecutePipeline)` callback
    that wraps the relevant MIR ops at materialization time, and
  - one `@lowering(target=iir.cf, source=<WrapOp>)` rule that emits
    the IIR specialization the legacy `if ctx.<flag>:` branch
    produced.

Importing the submodules has the side effect of registering all
three. Wiring happens in the parent dialect package's
`_register_passes()` (see `sorted_array/__init__.py`) so the
registrations land at the same dialect-import boundary as the
existing `lower_scan_pipeline` registration.
'''
