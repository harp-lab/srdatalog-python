'''Public compilation errors, independent of generated C++ spelling.'''

import pytest
from test_bitmap_plan import projection

from srdatalog.compile import compile_pipeline
from srdatalog.dsl import Program
from srdatalog.ir.codegen.cuda.batchfile import _collect_pipelines
from srdatalog.ir.hir import compile_to_mir


def _projection_pipeline(*, bitmap=False):
  join, value, destination, assign, points, output = projection()
  rule = (output(value, destination) <= assign(join, destination) & points(join, value)).with_plan(
    dedup_bitmap=bitmap
  )
  return _collect_pipelines(compile_to_mir(Program([rule])))[0]


def test_compile_pipeline_rejects_unknown_target():
  with pytest.raises(ValueError):
    compile_pipeline(_projection_pipeline(), target='cpp_tbb')  # type: ignore[arg-type]


def test_bitmap_plan_requires_a_complete_runner():
  with pytest.raises(ValueError):
    compile_pipeline(_projection_pipeline(bitmap=True))
