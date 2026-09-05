'''Count-only orchestrator regressions.

Count runners intentionally omit tail-mode ``execute_fused`` and do not
materialize their destination relation. The host orchestrator must therefore
stay on the count/readback path and retain that result for the C ABI.
'''

from srdatalog.ir.codegen.cuda.main_file import (
  _gen_final_print_block,
  gen_extern_c_shim,
  gen_runner_struct,
)
from srdatalog.ir.codegen.cuda.orchestrator import _gen_parallel_group
from srdatalog.ir.hir.types import RelationDecl, Version
from srdatalog.ir.mir import ExecutePipeline, InsertInto, ParallelGroup, Program


def _count_pipeline(rule_name: str, rel_name: str) -> ExecutePipeline:
  dest = InsertInto(rel_name=rel_name, version=Version.NEW, vars=['x'], index=[0])
  return ExecutePipeline(
    pipeline=[dest],
    source_specs=[],
    dest_specs=[dest],
    rule_name=rule_name,
    count=True,
  )


def test_count_only_parallel_group_never_calls_missing_fused_runner():
  group = ParallelGroup(
    ops=[
      _count_pipeline('CountA', 'OutA'),
      _count_pipeline('CountB', 'OutB'),
    ]
  )

  out = _gen_parallel_group(group, '    ', '0', {}, set())

  assert 'if (_tail_mode && false)' in out
  assert 'execute_fused' not in out
  assert 'JitRunner_CountA::launch_count' in out
  assert 'JitRunner_CountB::launch_count' in out


def test_count_only_parallel_group_records_destination_counts():
  group = ParallelGroup(
    ops=[
      _count_pipeline('CountA', 'OutA'),
      _count_pipeline('CountB', 'OutB'),
    ]
  )

  out = _gen_parallel_group(group, '    ', '0', {}, set())

  assert 'record_count_result("OutA", total_0);' in out
  assert 'record_count_result("OutB", total_1);' in out


def test_print_size_uses_retained_count_instead_of_empty_relation():
  decl = RelationDecl(
    rel_name='OutA',
    types=['int'],
    semiring='NoProvenance',
    print_size=True,
  )

  out = _gen_final_print_block([decl], {'OutA': [0]}, ['OutA'])

  assert 'get_count_result("OutA", count_result);' in out
  assert 'get_relation_by_schema<OutA' not in out


def test_runner_clears_retained_counts_at_start_of_each_run():
  decl = RelationDecl(rel_name='OutA', types=['int'], semiring='NoProvenance')
  mir = Program(steps=[(_count_pipeline('CountA', 'OutA'), False)])

  out = gen_runner_struct('CountPlan', [decl], mir, [''])

  run_start = out.index('static void run(')
  assert out.index('clear_count_results();', run_start) > run_start


def test_c_abi_retains_device_db_and_prefers_count_results():
  decl = RelationDecl(rel_name='OutA', types=['int'], semiring='NoProvenance')

  out = gen_extern_c_shim('CountPlan', [decl], ['OutA'])

  assert 'static CountPlan_DB_DeviceDB* g_device_db = nullptr;' in out
  assert 'CountPlan_Runner::get_count_result(rn, count_result)' in out
  assert 'get_relation_by_schema<OutA, FULL_VER>(*g_device_db).size()' in out
  assert 'unsigned long long srdatalog_dev_count(const char* rel_name)' in out
  assert 'unsigned long long srdatalog_dev_itemsize(const char* rel_name)' in out
  assert 'void* srdatalog_dev_ptr(const char* rel_name, unsigned col)' in out
  assert 'rel.unsafe_interned_columns().template column_ptr<0>()' in out
