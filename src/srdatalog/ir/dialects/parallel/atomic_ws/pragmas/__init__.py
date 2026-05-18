'''parallel.atomic_ws dialect pragmas.

Per `docs/phase_c_pragma_materialization.md` §4.2, the work-stealing
pragma lives under this sub-dialect's `pragmas/` subpackage because
its materialization produces ops that the sub-dialect owns
(`mir.WSScope` -> WS-flagged IIR emission). Each module here
defines:

  - one `Pragma` subclass (typed compile-time object, per
    `docs/pragma_as_typed_object.md` §2),
  - one `@pragma_handler(PragmaCls, on=mir.ExecutePipeline)` callback
    that wraps the relevant MIR ops at materialization time, and
  - one `@lowering(target=iir.cf, source=<WrapOp>)` rule that emits
    the IIR specialization the legacy `if ctx.<flag>:` branch
    produced.

Importing the submodules has the side effect of registering all
three. Wiring happens in the parent dialect package's
`_register_passes()` (see `atomic_ws/__init__.py`) so the
registrations land at the same dialect-import boundary as any other
sub-dialect pass.
'''
