'''Tests for codegen/jit/pipeline.py + codegen/jit/kernel_functor.py.

These are the top-level glue that ties root + nested emitters together.
'''
import sys
from pathlib import Path


import srdatalog.mir.types as m
from srdatalog.hir.types import Version
from srdatalog.codegen.jit.context import new_code_gen_context
from srdatalog.codegen.jit.pipeline import jit_nested_pipeline, jit_pipeline
from srdatalog.codegen.jit.kernel_functor import (
  jit_functor_start, jit_functor_end, jit_kernel_declaration,
  jit_full_kernel, jit_kernel_full,
)


def _cs(rel, ver, idx, prefix=(), handle_start=0):
  return m.ColumnSource(
    rel_name=rel, version=ver, index=idx,
    prefix_vars=list(prefix), handle_start=handle_start,
  )


# -----------------------------------------------------------------------------
# jit_functor_start / jit_functor_end / jit_kernel_declaration
# -----------------------------------------------------------------------------

def test_functor_start_warp_mode():
  out = jit_functor_start("Triangle")
  assert "// WARP MODE:" in out
  assert "struct Kernel_Triangle {" in out
  assert "static constexpr int kGroupSize = 32;" in out
  assert "__device__ void operator()(" in out


def test_functor_start_scalar_mode():
  out = jit_functor_start("Triangle", scalar_mode=True)
  assert "// SCALAR MODE:" in out
  assert "static constexpr int kGroupSize = 1;" in out


def test_functor_end():
  assert jit_functor_end() == "  }\n};\n"


def test_kernel_declaration():
  out = jit_kernel_declaration("Foo")
  assert "struct Kernel_Foo {" in out
  assert "operator()(" in out and ") const;" in out
  # No body: no opening brace for operator
  body_lines = [l for l in out.splitlines() if "const {" in l]
  assert not body_lines


# -----------------------------------------------------------------------------
# jit_nested_pipeline — empty / basic op dispatch
# -----------------------------------------------------------------------------

def test_nested_pipeline_empty_returns_empty():
  ctx = new_code_gen_context()
  assert jit_nested_pipeline([], ctx) == ""


def test_nested_pipeline_empty_closes_tiled_ballot_block():
  ctx = new_code_gen_context()
  ctx.tiled_cartesian_ballot_done = True
  ctx.indent = 2
  out = jit_nested_pipeline([], ctx)
  assert "warp_local_count += _tc_active;" in out
  # Ballot flag is reset
  assert ctx.tiled_cartesian_ballot_done is False


def test_nested_pipeline_scan_then_insert():
  ops = [
    m.Scan(vars=["x"], rel_name="R", version=Version.FULL,
           index=[0], handle_start=0),
    m.InsertInto(rel_name="S", version=Version.NEW, vars=["x"], index=[0]),
  ]
  ctx = new_code_gen_context()
  out = jit_nested_pipeline(ops, ctx)
  # Scan emits its loop
  assert "for (auto scan_it_" in out
  # InsertInto emits after
  assert ".emit_direct" in out


def test_nested_pipeline_filter_then_insert():
  ops = [
    m.Filter(vars=["x"], code="return x > 0;"),
    m.InsertInto(rel_name="R", version=Version.NEW, vars=["x"], index=[0]),
  ]
  ctx = new_code_gen_context()
  out = jit_nested_pipeline(ops, ctx)
  assert "if (x > 0)" in out
  assert "emit_direct" in out


def test_nested_pipeline_constant_bind_then_insert():
  ops = [
    m.ConstantBind(var_name="v", code="x + 1", deps=["x"]),
    m.InsertInto(rel_name="R", version=Version.NEW, vars=["v"], index=[0]),
  ]
  ctx = new_code_gen_context()
  out = jit_nested_pipeline(ops, ctx)
  assert "auto v = x + 1;" in out
  assert "emit_direct(v);" in out


def test_nested_pipeline_cj_pre_registers_child_handle_keys():
  '''Before recursing into rest, a nested ColumnJoin pre-registers child
  state keys so nested ops can find the child handles.'''
  cj = m.ColumnJoin(var_name="z", sources=[
    _cs("R", Version.FULL, [0, 1], prefix=("x",), handle_start=0),
  ])
  insert = m.InsertInto(rel_name="S", version=Version.NEW, vars=["z"], index=[0])
  ctx = new_code_gen_context()
  # Body should see the child name ch_R_0_z
  out = jit_nested_pipeline([cj, insert], ctx)
  assert "ch_R_0_z" in out


def test_nested_pipeline_cartesian_sets_inside_flag_for_rest():
  cart = m.CartesianJoin(
    vars=["x"],
    sources=[_cs("R", Version.FULL, [0], handle_start=0)],
    var_from_source=[["x"]],
  )
  # After Cartesian, an InsertInto with no Cartesian guard should NOT
  # emit lane-0 check (since we're inside Cartesian now).
  insert = m.InsertInto(rel_name="S", version=Version.NEW, vars=["x"], index=[0])
  ctx = new_code_gen_context()
  out = jit_nested_pipeline([cart, insert], ctx)
  # InsertInto inside Cartesian -> no `thread_rank() == 0` guard
  assert "thread_rank() == 0" not in out
  # Restored on return
  assert ctx.inside_cartesian is False
  assert ctx.cartesian_bound_vars == []


def test_nested_pipeline_cartesian_pre_narrows_following_negation():
  '''Nim: Cartesian + subsequent Negation triggers neg_pre_narrow
  registration for any prefix vars NOT already cartesian-bound.'''
  cart = m.CartesianJoin(
    vars=["x"],
    sources=[_cs("R", Version.FULL, [0], handle_start=0)],
    var_from_source=[["x"]],
  )
  # Negation uses p (not bound by cart) and x (bound by cart)
  neg = m.Negation(rel_name="N", version=Version.FULL, index=[0, 1],
                   prefix_vars=["p", "x"], handle_start=1)
  insert = m.InsertInto(rel_name="S", version=Version.NEW, vars=["x"], index=[0])
  ctx = new_code_gen_context()
  # Pre-bind p so neg's pre_vars=["p"] is applicable
  ctx.bound_vars.append("p")
  out = jit_nested_pipeline([cart, neg, insert], ctx)
  # Pre-narrow path: negation comment "// Using pre-narrowed handle"
  assert "Using pre-narrowed handle" in out


def test_nested_pipeline_cartesian_as_product_set_when_counting_safe():
  '''Counting + only InsertInto after Cartesian -> cartesian_as_product
  flag set before Cartesian body renders.'''
  cart = m.CartesianJoin(
    vars=["x"],
    sources=[_cs("R", Version.FULL, [0], handle_start=0)],
    var_from_source=[["x"]],
  )
  insert = m.InsertInto(rel_name="S", version=Version.NEW, vars=["x"], index=[0])
  ctx = new_code_gen_context()
  ctx.is_counting = True
  # Verify by mutating context — use a stub insert emitter isn't practical
  # here; rely on the flag being toggled back (via finally) after emit.
  # Before the call, flag is False.
  assert ctx.cartesian_as_product is False
  jit_nested_pipeline([cart, insert], ctx)
  # Restored after emit
  assert ctx.cartesian_as_product is False


def test_nested_pipeline_rejects_ws_flag_for_cj():
  cj = m.ColumnJoin(var_name="z", sources=[_cs("R", Version.FULL, [0, 1])])
  ctx = new_code_gen_context()
  ctx.ws_enabled = True
  try:
    jit_nested_pipeline([cj], ctx)
  except NotImplementedError as e:
    assert "work-stealing" in str(e)
  else:
    raise AssertionError("expected NotImplementedError")


# -----------------------------------------------------------------------------
# jit_pipeline — top-level entry
# -----------------------------------------------------------------------------

def test_pipeline_empty_returns_empty():
  ctx = new_code_gen_context()
  assert jit_pipeline([], [], ctx) == ""


def test_pipeline_emits_view_decls_then_root_scan():
  sc = m.Scan(vars=["x"], rel_name="Edge", version=Version.FULL,
              index=[0], handle_start=0)
  insert = m.InsertInto(rel_name="R", version=Version.NEW, vars=["x"], index=[0])
  ctx = new_code_gen_context()
  out = jit_pipeline([sc, insert], [], ctx)
  # Dedup view decl block emitted at top
  assert "using ViewType = std::remove_cvref_t<decltype(views[0])>;" in out
  assert "using HandleType = ViewType::NodeHandle;" in out
  assert "auto view_Edge_0_FULL_VER = views[0];" in out
  # Root scan emitted
  assert "// Root Scan: Edge" in out
  # Nested InsertInto emitted
  assert "emit_direct" in out


def test_pipeline_root_cj_multi_preregisters_handles():
  cj = m.ColumnJoin(var_name="z", sources=[
    _cs("R", Version.FULL, [0, 1], handle_start=0),
    _cs("S", Version.FULL, [1, 0], handle_start=1),
  ])
  insert = m.InsertInto(rel_name="T", version=Version.NEW, vars=["z"], index=[0])
  ctx = new_code_gen_context()
  out = jit_pipeline([cj, insert], [], ctx)
  # Root CJ emitted
  assert "// Root ColumnJoin (multi-source intersection)" in out
  # Pre-registered handles visible in state_key table during emit
  # (verify via presence of the deterministic names h_R_0_root / h_S_1_root)
  assert "h_R_0_root" in out
  assert "h_S_1_root" in out


# -----------------------------------------------------------------------------
# jit_full_kernel / jit_kernel_full — full functor envelope
# -----------------------------------------------------------------------------

def test_full_kernel_wraps_pipeline_in_struct_envelope():
  '''Full kernel emit = banner + struct + pipeline body + close.'''
  sc = m.Scan(vars=["x"], rel_name="Edge", version=Version.FULL,
              index=[0], handle_start=0)
  insert = m.InsertInto(rel_name="R", version=Version.NEW, vars=["x"], index=[0])
  ctx = new_code_gen_context()
  out = jit_full_kernel("MyRule", [sc, insert], ctx)
  assert "// JIT-Generated Kernel Functor: MyRule" in out
  assert "struct Kernel_MyRule {" in out
  assert "static constexpr int kBlockSize = 256;" in out
  assert "// WARP MODE:" in out
  assert "__device__ void operator()(" in out
  # Pipeline content
  assert "using ViewType = std::remove_cvref_t<decltype(views[0])>;" in out
  assert "// Root Scan: Edge" in out
  # Envelope close
  assert out.rstrip().endswith("};")


def test_kernel_full_on_execute_pipeline_node():
  sc = m.Scan(vars=["x"], rel_name="R", version=Version.FULL,
              index=[0], handle_start=0)
  insert = m.InsertInto(rel_name="S", version=Version.NEW, vars=["x"], index=[0])
  ep = m.ExecutePipeline(
    pipeline=[sc, insert],
    source_specs=[sc],
    dest_specs=[insert],
    rule_name="Foo",
  )
  out = jit_kernel_full(ep)
  assert "struct Kernel_Foo {" in out


# -----------------------------------------------------------------------------
# Goal-line probe: gen_jit_code Triangle rule through jit_full_kernel
# -----------------------------------------------------------------------------

def test_gen_jit_code_triangle_emits_through_full_pipeline():
  '''Not a byte-match assertion — just proves the full pipeline runs
  end-to-end on a real program without raising, and produces the key
  structural bits. The goal-line byte-match test lives separately.'''
  from test_integration_gen_jit_code import build_gen_jit_code
  from srdatalog.hir import compile_to_mir
  from srdatalog.codegen.batchfile import _collect_pipelines
  mir = compile_to_mir(build_gen_jit_code())
  pipelines = _collect_pipelines(mir)
  assert pipelines, "gen_jit_code should produce at least one pipeline"

  ep = pipelines[0]
  assert ep.rule_name == "Triangle"
  out = jit_kernel_full(ep)
  # Functor envelope
  assert "struct Kernel_Triangle {" in out
  # View decls (2 unique Edge views: 0,1 and 1,0 FULL)
  assert "view_Edge_0_1_FULL_VER" in out
  assert "view_Edge_1_0_FULL_VER" in out
  # Root emission happens (either root CJ or root Scan — depends on lowering)
  assert "// Root" in out


if __name__ == "__main__":
  import inspect
  this = sys.modules[__name__]
  passed = 0
  failed = 0
  for name, fn in inspect.getmembers(this, inspect.isfunction):
    if not name.startswith("test_"):
      continue
    try:
      fn()
      passed += 1
    except Exception as e:
      failed += 1
      print(f"FAIL {name}: {e}")
  print(f"{passed} passed / {failed} fail")
