'''JIT codegen — Python port of src/srdatalog/codegen/target_jit/.

Emits the same CUDA/C++ text that Nim's JIT backend does
(genStepBody / genJitFileContentFromExecutePipeline), byte-matched
against Nim fixtures in python/tests/fixtures/jit/.

Submodule roadmap (following the Nim layout):

  plugin        — index-type plugin registry + default DSAI hooks
                  (port of index_plugin.nim)
  context       — CodeGenContext, CodeGenHooks, RunnerGenState,
                  sanitize_var_name, ind/inc_indent/dec_indent,
                  plugin-dispatched gen_* C++ expression helpers
                  (port of jit_base.nim)
  emit_helpers  — jit_filter / jit_constant_bind / jit_insert_into,
                  has_balanced_scan, get_balanced_scan_info,
                  count_handles_in_pipeline
                  (port of jit_emit_helpers.nim)
  [coming next] view_management, scan_negation, instructions, root,
                pipeline, complete_runner, ws_kernel, materialized,
                orchestrator, file
'''
