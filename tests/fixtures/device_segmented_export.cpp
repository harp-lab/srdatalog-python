#include "srdatalog.h"
#include "gpu/device_2level_index.h"
#include "gpu/runtime/query.h"
#include <rmm/mr/device/limiting_resource_adaptor.hpp>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace SRDatalog;
using namespace SRDatalog::AST;
using namespace SRDatalog::AST::Literals;

#include "generated_relation_export.h"

namespace {
using ExportRows = RelationSchema<decltype("ExportRows"_s), NoProvenance,
    std::tuple<int, int, int>, SRDatalog::GPU::Device2LevelIndex>;
using Blueprint = Database<ExportRows>;
using DeviceDB = SemiNaiveDatabase<Blueprint, SRDatalog::GPU::DeviceRelationType>;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template<class Rel>
void fill(Rel& relation, const std::vector<uint32_t>& ids, const IndexSpec& canonical) {
  std::array<std::vector<uint32_t>, 3> host;
  for (auto& column : host) column.reserve(ids.size());
  for (const auto id : ids) {
    host[0].push_back(static_cast<uint32_t>(static_cast<int>(id) - 70000));
    host[1].push_back(id % 7);
    host[2].push_back(2 * id + 1);
  }
  auto& columns = relation.unsafe_interned_columns();
  columns.resize(ids.size());
  for (std::size_t column = 0; column < host.size(); ++column) {
    srdatalog_check_gpu(GPU_MEMCPY(columns.column_ptr(column), host[column].data(),
                                  ids.size() * sizeof(uint32_t), GPU_HOST_TO_DEVICE));
  }
  relation.build_index_take_ownership(canonical);
}
}  // namespace

extern "C" int run_device_segmented_export_regression(const char* path) {
  try {
    DeviceDB db;
    auto& full = get_relation_by_schema<ExportRows, FULL_VER>(db);
    auto& newt = get_relation_by_schema<ExportRows, NEW_VER>(db);
    auto& delta = get_relation_by_schema<ExportRows, DELTA_VER>(db);
    const IndexSpec canonical{2, 0, 1};
    constexpr std::size_t base_rows = 65539;
    constexpr std::size_t head_rows = 65543;
    std::vector<uint32_t> base;
    std::vector<uint32_t> candidates;
    for (std::size_t row = 0; row < base_rows; ++row) base.push_back(2 * row);
    candidates = base;  // Existing FULL rows must not reappear in HEAD.
    for (std::size_t row = 0; row < head_rows; ++row) {
      candidates.push_back(2 * row + 1);
      candidates.push_back(2 * row + 1);  // Repeated derivations must deduplicate.
    }
    fill(full, base, canonical);
    fill(newt, candidates, canonical);
    delta.ensure_index(canonical, false);
    auto& index = full.get_index(canonical);
    auto& delta_index = delta.get_index(canonical);
    newt.get_index(canonical).set_difference_update(index, delta_index);
    newt.release_device_storage();
    index.merge(delta_index, 0);  // Populate HEAD, deliberately do not compact.
    require(index.full().size() == base_rows && index.head().size() == head_rows,
            "fixture did not produce disjoint nonempty base and HEAD");
    require(index.size() == base_rows + head_rows, "wrong logical segmented cardinality");
    const auto* base_pointer = index.full().data().data();
    const auto* head_pointer = index.head().data().data();
    {
      // Export must succeed without allocating any new device storage. The old
      // compact-before-export implementation fails under this resource limit.
      auto* previous = rmm::mr::get_current_device_resource();
      rmm::mr::limiting_resource_adaptor<rmm::mr::device_memory_resource> no_alloc(previous, 0);
      struct RestoreResource {
        rmm::mr::device_memory_resource* previous;
        ~RestoreResource() { rmm::mr::set_current_device_resource(previous); }
      } restore{previous};
      rmm::mr::set_current_device_resource(&no_alloc);
      srdatalog_write_tsv<ExportRows>(db, path, canonical);
    }
    require(index.full().size() == base_rows && index.head().size() == head_rows &&
                index.full().data().data() == base_pointer &&
                index.head().data().data() == head_pointer,
            "TSV export consumed or compacted its source segments");
    require(delta_index.size() == head_rows, "TSV export consumed independent DELTA");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
