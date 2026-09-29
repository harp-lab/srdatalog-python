#include "gpu/device_2level_index.h"
#include "relation_col.h"

#include <cstdio>
#include <exception>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using Pair = std::tuple<uint32_t, uint32_t>;
using Rel = SRDatalog::Relation<NoProvenance, Pair, SRDatalog::GPU::Device2LevelIndex,
                               SRDatalog::DeviceRelationPolicy>;
using Stats = rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>;

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void check_cuda(cudaError_t result) {
  if (result != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(result));
  }
}

void fill(Rel& relation, const std::vector<uint32_t>& values) {
  auto& columns = relation.unsafe_interned_columns();
  columns.resize(values.size());
  check_cuda(cudaMemset(columns.column_ptr(0), 0, values.size() * sizeof(uint32_t)));
  check_cuda(cudaMemcpy(columns.column_ptr(1), values.data(), values.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  relation.build_index_take_ownership(SRDatalog::IndexSpec{0, 1});
}

std::vector<uint32_t> read_values(const Rel::IndexTypeInst& index) {
  std::vector<uint32_t> keys(index.size());
  std::vector<uint32_t> values(index.size());
  check_cuda(cudaMemcpy(keys.data(), index.data().column_ptr(0),
                        keys.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  check_cuda(cudaMemcpy(values.data(), index.data().column_ptr(1),
                        values.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  for (const auto key : keys) {
    require(key == 0, "release or compaction changed tuple keys");
  }
  return values;
}

void check_release_and_recursive_reuse(Stats& stats) {
  const SRDatalog::IndexSpec spec{0, 1};
  Rel full;
  Rel newt;
  Rel delta;
  std::vector<uint32_t> frontier(2048);
  for (std::size_t i = 0; i < frontier.size(); ++i) {
    frontier[i] = i;
  }
  fill(full, frontier);

  // Ordinary clear must still retain capacity; the next consuming build swaps
  // this old index allocation into NEWT's empty intern-column storage.
  fill(newt, frontier);
  newt.clear();
  require(newt.get_index(spec).full().data().capacity() >= frontier.size(),
          "ordinary clear unexpectedly released reusable index storage");

  for (std::size_t iteration = 0; iteration < 2; ++iteration) {
    std::vector<uint32_t> candidates;
    for (const auto value : frontier) {
      candidates.push_back(2 * value);
      candidates.push_back(2 * value + 1);
    }
    fill(newt, candidates);
    auto& new_index = newt.get_index(spec);
    auto& full_index = full.get_index(spec);
    delta.clear();
    delta.ensure_index(spec, false);
    auto& delta_index = delta.get_index(spec);
    new_index.set_difference_update(full_index, delta_index);

    const auto dead_rows = new_index.full().data().capacity() +
                           newt.unsafe_interned_columns().capacity();
    if (iteration == 0) {
      require(newt.unsafe_interned_columns().capacity() >= 2048,
              "fixture did not retain the old intern-column allocation");
    }
    const auto before_release = stats.get_bytes_counter().value;
    newt.release_device_storage();
    require(before_release - stats.get_bytes_counter().value >=
                static_cast<int64_t>(dead_rows * 2 * sizeof(uint32_t)),
            "last-use cleanup retained dead NEWT device allocations");
    require(newt.unsafe_interned_columns().capacity() == 0,
            "released NEWT retained intern-column capacity");
    require(newt.has_index(spec) && !newt.is_dirty(spec),
            "release lost index registration or left dirty empty metadata");
    require(newt.get_index(spec).empty(), "released NEWT still exposes old tuples");

    const auto expected_begin = iteration == 0 ? 2048U : 4096U;
    const auto expected_end = iteration == 0 ? 4096U : 8192U;
    std::vector<uint32_t> expected_delta;
    for (uint32_t value = expected_begin; value < expected_end; ++value) {
      expected_delta.push_back(value);
    }
    require(read_values(delta_index) == expected_delta,
            "NEWT release invalidated the live recursive DELTA");

    full_index.merge(delta_index, 0);
    const auto before_compact = stats.get_bytes_counter().value;
    full_index.compact();
    // FULL+HEAD and compacted FULL contain identical disjoint tuples. Allow
    // small alignment/root metadata differences, not a retained 16+KiB HEAD.
    require(stats.get_bytes_counter().value <= before_compact + 4096,
            "compaction retained obsolete HEAD device allocations");
    require(full_index.head().empty(), "compaction left a visible HEAD segment");
    std::vector<uint32_t> expected_full(expected_end);
    for (uint32_t value = 0; value < expected_end; ++value) {
      expected_full[value] = value;
    }
    require(read_values(full_index) == expected_full, "recursive FULL tuple set is incorrect");
    frontier = read_values(delta_index);
    require(frontier == expected_delta, "compaction consumed live recursive DELTA");
  }

  using AnnotatedRel = SRDatalog::Relation<NaturalBag, Pair,
      SRDatalog::GPU::Device2LevelIndex, SRDatalog::DeviceRelationPolicy>;
  AnnotatedRel annotated;
  const auto before_annotations = stats.get_bytes_counter().value;
  annotated.unsafe_interned_columns().resize(2048);
  annotated.provenance().resize(2048);
  annotated.release_device_storage();
  require(annotated.provenance().capacity() == 0 &&
              stats.get_bytes_counter().value == before_annotations,
          "last-use cleanup retained provenance storage");
}
}  // namespace

extern "C" int run_device_storage_release_regression() {
  try {
    auto* previous = rmm::mr::get_current_device_resource();
    Stats stats(previous);
    struct RestoreResource {
      rmm::mr::device_memory_resource* previous;
      ~RestoreResource() { rmm::mr::set_current_device_resource(previous); }
    } restore{previous};
    rmm::mr::set_current_device_resource(&stats);
    std::exception_ptr failure;
    // Join before destroying the accounting resource, including destruction of
    // any per-thread runtime scratch that captured the resource during build.
    std::thread worker([&] {
      try {
        check_cuda(cudaSetDevice(0));
        check_release_and_recursive_reuse(stats);
      } catch (...) {
        failure = std::current_exception();
      }
    });
    worker.join();
    if (failure) {
      std::rethrow_exception(failure);
    }
    require(stats.get_bytes_counter().value == 0, "regression leaked device allocations");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
