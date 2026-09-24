'''Execute-only CUDA runner for exact binary projection bitmap plans.'''

from __future__ import annotations

import srdatalog.ir.mir.types as m
from srdatalog.ir.hir.types import Version


def _bind_relation(source: m.ColumnSource, name: str) -> str:
  full = f'get_relation_by_schema<{source.rel_name}, FULL_VER>(db)'
  if source.version is Version.DELTA:
    relation = (
      f'(iteration == 0) ? {full} : get_relation_by_schema<{source.rel_name}, DELTA_VER>(db)'
    )
  elif source.version is Version.FULL:
    relation = full
  else:
    raise ValueError('dedup_bitmap sources must use FULL or DELTA')
  cols = ', '.join(map(str, source.index))
  return (
    f'  auto& {name}_relation = {relation};\n'
    f'  const auto& {name}_index = {name}_relation.get_index(SRDatalog::IndexSpec{{{{{cols}}}}});\n'
  )


def gen_bitmap_runner(
  node: m.ExecutePipeline,
  db_type_name: str,
  rel_index_types: dict[str, str],
) -> tuple[str, str]:
  '''Emit an exact set-projection runner, without hash or ordinary join kernels.'''
  plan = node.bitmap_join
  if plan is None or len(node.dest_specs) != 1:
    raise ValueError('dedup_bitmap requires one binary destination')
  if node.count or node.dedup_hash or node.work_stealing or node.block_group or node.use_fan_out:
    raise ValueError('dedup_bitmap cannot be combined with other execution strategies')
  dest = node.dest_specs[0]
  if dest.version is not Version.NEW or sorted(dest.index) != [0, 1] or len(dest.vars) != 2:
    raise ValueError('dedup_bitmap requires a binary NEW destination')
  for source in (plan.assign, plan.points):
    if sorted(source.index) != [0, 1] or source.prefix_vars:
      raise ValueError('dedup_bitmap requires unconstrained binary source indexes')
    index_type = rel_index_types.get(source.rel_name, '')
    if index_type and not any(
      t in index_type for t in ('DeviceSortedArrayIndex', 'Device2LevelIndex')
    ):
      raise ValueError(f'dedup_bitmap does not support index type {index_type!r}')

  runner = f'JitRunner_{node.rule_name}'
  declaration = (
    f'struct {runner} {{\n'
    f'  using DB = {db_type_name};\n'
    '  static void execute(DB& db, uint32_t iteration);\n'
    '};\n\n'
  )
  dictionary = ', '.join(map(str, reversed(plan.points.index)))
  body = f'''void {runner}::execute(DB& db, uint32_t iteration) {{
  nvtxRangePushA("{node.rule_name}");
  struct RangeEnd {{ ~RangeEnd() {{ nvtxRangePop(); }} }} range_end;
  namespace bitmap = SRDatalog::GPU::bitmap;
  bitmap::Input input{{}};
  auto columns = [](const auto& index) -> bitmap::Columns {{
    using Index = std::remove_cvref_t<decltype(index)>;
    static_assert(Index::arity == 2 && std::is_same_v<typename Index::ValueType, uint32_t>);
    if (index.size() == 0) return {{}};
    return {{index.size(), index.data().template column_ptr<0>(),
            index.data().template column_ptr<1>()}};
  }};
  auto segments = [&](const auto& index, auto& full, auto& head) {{
    if constexpr (requires {{ index.full(); index.head(); }}) {{
      full = columns(index.full());
      head = columns(index.head());
    }} else {{
      full = columns(index);
    }}
  }};
  auto keys = [](const auto& index) -> bitmap::Keys {{
    if (index.size() == 0) return {{}};
    if (index.num_unique_root_values() == 0)
      throw std::runtime_error("dedup_bitmap: nonempty index has no value-key cache");
    return {{index.num_unique_root_values(), index.root_unique_values().data()}};
  }};
'''
  body += _bind_relation(plan.assign, 'assign')
  body += _bind_relation(plan.points, 'points')
  body += f'''  segments(assign_index, input.assign, input.assign_head);
  segments(points_index, input.points_full, input.points_head);
  const auto& value_index = points_relation.get_index(SRDatalog::IndexSpec{{{{{dictionary}}}}});
  auto collect_keys = [&](const auto& index) {{
    if constexpr (requires {{ index.full(); index.head(); }}) {{
      input.heaps_full = keys(index.full());
      input.heaps_head = keys(index.head());
    }} else {{
      input.heaps_full = keys(index);
    }}
  }};
  collect_keys(value_index);
  auto& destination = get_relation_by_schema<{dest.rel_name}, NEW_VER>(db);
  using Destination = std::remove_reference_t<decltype(destination)>;
  static_assert(!has_provenance_v<typename Destination::semiring_type>);
  bitmap::execute(input, [&](uint64_t rows) -> bitmap::Output {{
    const uint64_t old_rows = destination.size();
    constexpr uint64_t limit = std::numeric_limits<uint32_t>::max();
    if (old_rows > limit || rows > limit - old_rows)
      throw std::overflow_error("dedup_bitmap: NEW relation exceeds uint32 row limit");
    destination.resize_interned_columns(static_cast<std::size_t>(old_rows + rows), 0);
    return {{destination.template interned_column<0>() + old_rows,
            destination.template interned_column<1>() + old_rows}};
  }});
}}

'''
  return declaration, '#include "gpu/runtime/jit/bitmap_join.h"\n\n' + declaration + body
