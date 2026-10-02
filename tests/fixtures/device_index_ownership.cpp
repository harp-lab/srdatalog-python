#include "gpu/device_sorted_array_index.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

using namespace SRDatalog;
using namespace SRDatalog::GPU;

namespace {
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

using Row = std::array<uint32_t, 4>;
using Attrs = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>;

template <typename SR>
void check_build(const std::array<int, 4>& order, std::size_t count,
                 bool supply_provenance = true) {
  NDDeviceArray<uint32_t, 4> columns;
  columns.reserve(count + 17);  // The consumed allocation need not have stride == rows.
  columns.resize(count);
  DeviceArray<semiring_value_t<SR>> provenance;
  std::vector<uint64_t> annotations(count);
  std::array<std::vector<uint32_t>, 4> input;
  for (auto& col : input) {
    col.resize(count);
  }
  std::map<Row, uint64_t> expected;
  for (std::size_t row = 0; row < count; ++row) {
    const auto id = static_cast<uint32_t>((row * 73) % 257);
    Row source{id, 500 - id, id % 11, (id * 17) % 257};
    Row key;
    for (std::size_t col = 0; col < 4; ++col) {
      input[col][row] = source[col];
      key[col] = source[order[col]];
    }
    annotations[row] = row + 1;
    expected[key] += supply_provenance ? annotations[row] : 1;
  }
  for (std::size_t col = 0; col < 4; ++col) {
    if (count != 0) {
      check_cuda(cudaMemcpy(columns.column_ptr(col), input[col].data(),
                            count * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
  }
  if constexpr (has_provenance_v<SR>) {
    if (supply_provenance && count != 0) {
      provenance.resize(count);
      check_cuda(cudaMemcpy(provenance.data(), annotations.data(),
                            count * sizeof(uint64_t), cudaMemcpyHostToDevice));
    }
  }

  const auto* original_allocation = columns.data();
  DeviceSortedArrayIndex<SR, Attrs> index;
  index.build_take_ownership(IndexSpec{order[0], order[1], order[2], order[3]},
                             columns, provenance);
  check_cuda(cudaDeviceSynchronize());
  require(index.size() == expected.size(), "wrong deduplicated tuple count");
  require(columns.num_rows() == 0, "full-arity build did not consume input columns");
  require(index.rows_processed() == 0, "consumed index appears dirty");
  if constexpr (has_provenance_v<SR>) {
    require(provenance.size() == 0, "matching input provenance was not consumed");
  }
  if (count == 0) {
    require(!index.root().valid(), "empty input has a valid root");
    return;
  }
  if constexpr (!has_provenance_v<SR>) {
    require(index.data().data() == original_allocation,
            "full-arity build allocated replacement relation storage");
  }
  std::array<std::vector<uint32_t>, 4> output;
  for (std::size_t col = 0; col < 4; ++col) {
    output[col].resize(index.size());
    check_cuda(cudaMemcpy(output[col].data(), index.data().column_ptr(col),
                          index.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  }
  std::vector<uint64_t> output_provenance(index.size());
  if constexpr (has_provenance_v<SR>) {
    check_cuda(cudaMemcpy(output_provenance.data(), index.provenance_ptr().get(),
                          index.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  }
  std::size_t row = 0;
  for (const auto& [tuple, annotation] : expected) {
    for (std::size_t col = 0; col < 4; ++col) {
      require(output[col][row] == tuple[col], "column order or tuple contents changed");
    }
    if constexpr (has_provenance_v<SR>) {
      require(output_provenance[row] == annotation,
              "row permutation or duplicate aggregation corrupted provenance");
    }
    ++row;
  }
}

template <typename SR>
void check_root_key_capacity() {
  constexpr std::size_t count = 32768;
  using Pair = std::tuple<uint32_t, uint32_t>;
  NDDeviceArray<uint32_t, 2> columns(count);
  DeviceArray<semiring_value_t<SR>> provenance;
  DeviceSortedArrayIndex<SR, Pair> index;
  std::vector<uint32_t> first(count);
  std::vector<uint32_t> second(count);
  for (std::size_t row = 0; row < count; ++row) {
    second[row] = row;
  }

  // A fresh many-tuples/few-roots build, followed by growing and then shrinking
  // root cardinality on the same index, must not retain row-sized root buffers.
  for (const std::size_t roots : {std::size_t{4}, count, std::size_t{1}}) {
    columns.resize(count);
    for (std::size_t row = 0; row < count; ++row) {
      first[row] = row / (count / roots);
    }
    check_cuda(cudaMemcpy(columns.column_ptr(0), first.data(), count * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(columns.column_ptr(1), second.data(), count * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    index.build_take_ownership(IndexSpec{0, 1}, columns, provenance);
    check_cuda(cudaDeviceSynchronize());
    require(index.size() == count, "root extraction dropped distinct full tuples");
    require(index.root_unique_values().size() == roots, "wrong distinct root count");
    require(index.root_unique_values().capacity() <= std::max(2 * roots, std::size_t{1024}),
            "root key capacity scales with rows instead of distinct roots");
    std::vector<uint32_t> observed_roots(roots);
    check_cuda(cudaMemcpy(observed_roots.data(), index.root_unique_values().data(),
                          roots * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for (std::size_t root = 0; root < roots; ++root) {
      require(observed_roots[root] == root, "root keys are missing or unordered");
    }
    std::vector<uint32_t> observed_rows(count);
    check_cuda(cudaMemcpy(observed_rows.data(), index.data().column_ptr(1),
                          count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    require(observed_rows == second, "root extraction changed full tuple contents");
  }

  columns.resize(0);
  generate_unique(columns, index.root_unique_values());
  require(index.root_unique_values().size() == 0, "empty root extraction retained stale keys");
}

__global__ void large_offset(std::size_t* output) {
  SRDatalog::GPU::NodeView<NoProvenance> view;
  view.stride_ = 1631555384U;
  SRDatalog::GPU::NodeHandle<NoProvenance> handle(0, 1, 3);
  *output = handle.offset<3>(17, view);
}
}  // namespace

extern "C" int run_device_index_ownership_regression() {
  try {
    std::array<int, 4> order{0, 1, 2, 3};
    do {
      check_build<NoProvenance>(order, 2056);
    } while (std::next_permutation(order.begin(), order.end()));
    check_build<NaturalBag>({2, 0, 3, 1}, 2056);
    check_build<NaturalBag>({0, 1, 2, 3}, 2056);
    check_build<NaturalBag>({2, 0, 3, 1}, 2056, false);
    check_build<NaturalBag>({2, 0, 3, 1}, 1);
    check_build<NoProvenance>({2, 0, 3, 1}, 0);
    check_root_key_capacity<NoProvenance>();
    check_root_key_capacity<NaturalBag>();
    DeviceArray<std::size_t> offset(1);
    large_offset<<<1, 1>>>(offset.data());
    check_cuda(cudaGetLastError());
    std::size_t observed = 0;
    check_cuda(cudaMemcpy(&observed, offset.data(), sizeof(observed), cudaMemcpyDeviceToHost));
    require(observed == std::size_t{1631555384} * 3 + 17,
            "flattened column offset wrapped at 32 bits");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
