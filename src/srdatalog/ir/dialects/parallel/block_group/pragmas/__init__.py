'''parallel.block_group dialect pragmas.

Per `docs/phase_c_pragma_materialization.md` §4.1, the `block_group`
pragma owns a NEW sub-dialect; this subpackage holds the typed
`Pragma` subclass + handler + the wrap-op lowering. Each module
defines:

  - one `Pragma` subclass (typed compile-time object, per
    `docs/pragma_as_typed_object.md` §2),
  - one `@pragma_handler(PragmaCls, on=mir.ExecutePipeline)` callback
    that wraps the relevant MIR root op at materialization time, and
  - one `@lowering(target=iir.cf, source=<WrapOp>)` rule that emits
    the IIR specialization the legacy `if ctx.bg_enabled:` branch
    produced.

Importing the submodules has the side effect of registering all
three. Wiring happens in the parent dialect package's
`_register_passes()` (see `parallel/block_group/__init__.py`) so the
registrations land at the same dialect-import boundary as other
dialects.
'''
