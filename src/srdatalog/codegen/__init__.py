'''C++ codegen backend (non-template API).

Consumes `mir_types.MirNode` IR directly — no parallel node hierarchy.
Ported from mhk's `python-api-notemplate` branch, adapted to share the
same MIR types as the HIR→MIR pipeline on this branch.

Submodules:
  schema      — FactDefinition / Pragma / SchemaDefinition (C++ prelude emission)
  cpp_emit    — free-function emitters keyed on MIR node type (to be added)
  helpers     — CodeGenContext and view-spec collection (to be added)
  batchfile   — JIT batch kernel (.cpp) generation (to be added)
  orchestrator — SRDatalogProgram + orchestrator (.cpp) generation (to be added)
'''
