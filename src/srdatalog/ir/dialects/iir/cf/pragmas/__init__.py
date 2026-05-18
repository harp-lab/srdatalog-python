'''iir.cf dialect pragmas.

Per `docs/phase_c_pragma_materialization.md` §4.3, pragmas that
specialize control-flow IIR emission (currently just `count`) live
under this dialect's `pragmas/` subpackage. Each module defines:

  - one `Pragma` subclass (typed compile-time object, per
    `docs/pragma_as_typed_object.md` §2),
  - one `@pragma_handler(PragmaCls, on=mir.ExecutePipeline)` callback
    that wraps the relevant MIR ops at materialization time, and
  - one `@lowering(target=iir.cf, source=<WrapOp>)` rule that emits
    the IIR specialization the legacy `is_counting` / `ep.count`
    branch produced.

Importing the submodules has the side effect of registering the
handler. Wiring of the lowering happens in the parent dialect
package's `_register_passes()` so the registration lands at the same
dialect-import boundary as the existing iir.cf verifier.
'''
