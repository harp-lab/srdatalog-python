/**
 * @file test_gpu_pipeline_executor_let_if.cpp
 * @brief Test GPU pipeline execution for LetClause and IfClause
 */

#define BOOST_TEST_MODULE GPUPipelineExecutorLetIfTest
#include <boost/test/included/unit_test.hpp>

#include <thrust/copy.h>
#include <thrust/fill.h>

#include "gpu/device_array.h"
#include "gpu/device_sorted_array_index.h"
#include "gpu/gpu_api.h"  // GPU API abstraction (replaces cuda_runtime.h)
#include "gpu/nd_device_array.h"
#include "gpu/runtime/runtime.h"
#include "mir.h"
#include "relation_col.h"

using namespace SRDatalog;
using namespace SRDatalog::AST::Literals;
using namespace SRDatalog::GPU;

// ============================================================================
// Helper Types & Functions
// ============================================================================

using BooleanSR = ::BooleanSR;

// 1-column index (for simple tests)
using IndexType1 = DeviceSortedArrayIndex<BooleanSR, std::tuple<uint32_t>, uint32_t>;

// 2-column index (standard)
using IndexType2 =
    DeviceSortedArrayIndex<BooleanSR, std::tuple<uint32_t, uint32_t>, uint32_t, uint32_t>;

// 3-column index (for LetClause result)
using IndexType3 =
    DeviceSortedArrayIndex<BooleanSR, std::tuple<uint32_t, uint32_t, uint32_t>, uint32_t, uint32_t>;

using HandleType1 = typename IndexType1::NodeHandle;
using HandleType2 = typename IndexType2::NodeHandle;
using HandleType3 = typename IndexType3::NodeHandle;

inline void device_sync() {
  GPU_ERROR_T err = GPU_DEVICE_SYNCHRONIZE();
  BOOST_REQUIRE_EQUAL(err, GPU_SUCCESS);
}

// Build 1-column index
void build_index_1(IndexType1& index, const std::vector<uint32_t>& values,
                   const std::vector<int>& col_order) {
  if (values.empty())
    return;
  std::size_t n = values.size();
  NDDeviceArray<uint32_t, 1> cols(n);
  DeviceArray<semiring_value_t<BooleanSR>> prov(n);

  GPU_MEMCPY(cols.column_ptr(0), values.data(), n * sizeof(uint32_t), GPU_HOST_TO_DEVICE);
  thrust::fill(thrust::device, prov.begin(), prov.end(), BooleanSR::one());

  IndexSpec spec;
  spec.cols = col_order;
  index.build_from_encoded_device(spec, cols, prov);
}

// Build 2-column index
void build_index_2(IndexType2& index, const std::vector<std::pair<uint32_t, uint32_t>>& edges,
                   const std::vector<int>& col_order) {
  if (edges.empty())
    return;
  std::size_t n = edges.size();
  NDDeviceArray<uint32_t, 2> cols(n);
  DeviceArray<semiring_value_t<BooleanSR>> prov(n);

  std::vector<uint32_t> col0(n);
  std::vector<uint32_t> col1(n);
  for (std::size_t i = 0; i < n; ++i) {
    col0[i] = edges[i].first;
    col1[i] = edges[i].second;
  }

  GPU_MEMCPY(cols.column_ptr(0), col0.data(), n * sizeof(uint32_t), GPU_HOST_TO_DEVICE);
  GPU_MEMCPY(cols.column_ptr(1), col1.data(), n * sizeof(uint32_t), GPU_HOST_TO_DEVICE);
  thrust::fill(thrust::device, prov.begin(), prov.end(), BooleanSR::one());

  IndexSpec spec;
  spec.cols = col_order;
  index.build_from_encoded_device(spec, cols, prov);
}

// Build 3-column index (empty for Z usually, but useful if we wanted to populate it)
void build_index_3_empty(IndexType3& index, const std::vector<int>& col_order) {
  // Empty build
  NDDeviceArray<uint32_t, 3> cols(0);
  DeviceArray<semiring_value_t<BooleanSR>> prov(0);
  IndexSpec spec;
  spec.cols = col_order;
  index.build_from_encoded_device(spec, cols, prov);
}

// ============================================================================
// Test Cases
// ============================================================================

BOOST_AUTO_TEST_CASE(test_gpu_let_clause) {
  std::cout << "=== Starting test_gpu_let_clause ===" << std::endl;
  using SR = BooleanSR;

  using RelationR = AST::RelationSchema<decltype("R"_s), SR, std::tuple<int, int>>;
  using RelationZ = AST::RelationSchema<decltype("Z"_s), SR, std::tuple<int, int, int>>;

  using x_ = AST::Var<decltype("x"_s)>;
  using y_ = AST::Var<decltype("y"_s)>;
  using sum_ = AST::Var<decltype("sum"_s)>;

  // R(x,y)
  using SourceR =
      mir::ColumnSource<mir::IndexSpecT<RelationR, std::integer_sequence<int, 0, 1>, FULL_VER>,
                        std::tuple<x_, y_>>;

  // Destination Z(x,y,sum)
  using IndexZ = mir::IndexSpecT<RelationZ, std::integer_sequence<int, 0, 1, 2>, FULL_VER>;
  using DestZ = mir::DestinationRelation<RelationZ, std::tuple<x_, y_, sum_>, NEW_VER, IndexZ>;

  // Let sum = x + y
  using LetSum =
      AST::LetClause<sum_,
                     AST::CppExpr<std::tuple<x_, y_>, [](int x, int y) -> int { return x + y; }>>;

  using SpecR_x = mir::IndexSpecT<RelationR, std::integer_sequence<int, 0>, FULL_VER>;
  using SourceR_x = mir::ColumnSource<SpecR_x, std::tuple<>>;
  // JoinX: HandleStart=0, VarToHandlesMap={} (no prefix vars)
  using JoinX = mir::ColumnJoin<x_, std::tuple<SourceR_x>, mir::DefaultJoinStrategy, 0,
                                decltype(hana::make_map())>;

  using SpecR_xy = mir::IndexSpecT<RelationR, std::integer_sequence<int, 0, 1>, FULL_VER>;
  using SourceR_y = mir::ColumnSource<SpecR_xy, std::tuple<x_>>;

  // JoinY: HandleStart=1 (after X), VarToHandlesMap={x -> 0} (x relies on handle 0)
  using VarToHandlesY = decltype(hana::make_map(
      hana::make_pair(hana::type_c<x_>, std::integer_sequence<std::size_t, 1>{})));
  using JoinY =
      mir::ColumnJoin<y_, std::tuple<SourceR_y>, mir::DefaultJoinStrategy, 1, VarToHandlesY>;

  using AllOps = std::tuple<JoinX, JoinY, LetSum, DestZ>;

  // Define Maps with CORRECT types
  using VarPosMap = decltype(hana::make_map(hana::make_pair(hana::type_c<x_>, hana::int_c<0>),
                                            hana::make_pair(hana::type_c<y_>, hana::int_c<1>),
                                            hana::make_pair(hana::type_c<sum_>, hana::int_c<2>)));

  using IteratorPosMap =
      decltype(hana::make_map(hana::make_pair(hana::type_c<SpecR_x>, hana::int_c<0>),
                              hana::make_pair(hana::type_c<SpecR_xy>, hana::int_c<1>)));

  using OutputFullIteratorPosMap =
      decltype(hana::make_map(hana::make_pair(hana::type_c<IndexZ>, hana::int_c<0>)));

  struct MyPipeline {
    using mir_ops_type = AllOps;
    using var_pos_map_type = VarPosMap;
    using iterator_pos_map_type = IteratorPosMap;
    using output_full_iterator_pos_map_type = OutputFullIteratorPosMap;
    using destination_specs_type = std::tuple<IndexZ>;
    using all_schemas_type = std::tuple<>;  // Phase 2 requirement
  };

  // Prepare Data using manual indices
  IndexType2 index_r_x;  // R indexed on {0, 1} but used for x (prefix {})
  // Actually SpecR_x is {0}. Index should be built on {0}.
  // But our build_index helper makes full index.
  // For prefix {}, we scan the root. The index structure matters for `get_index`.
  // Here we simulate the index.
  // R data: (1,2), (3,4), (10,20)

  build_index_2(index_r_x, {{1, 2}, {3, 4}, {10, 20}}, {0, 1});
  // Note: SpecR_x says {0}. But here we provided {0, 1}.
  // The Launcher passes handles. The handle should match what the executor expects.
  // Executor for JoinX uses `SourceR_x`. `IndexSpec` is {0}.
  // We can pass a handle that has more columns, but the executor iterates what it needs.
  // Wait, `ColumnSource` uses `IteratorPosMap` to find the handle.

  IndexType2 index_r_xy;
  build_index_2(index_r_xy, {{1, 2}, {3, 4}, {10, 20}}, {0, 1});

  IndexType3 index_z;
  build_index_3_empty(index_z, {0, 1, 2});

  // Get handles
  auto handle_r_x = index_r_x.root();
  auto handle_r_xy = index_r_xy.root();
  auto handle_z = index_z.root();

  // Pack handles manually (polymorphism via void* or template? Launcher expects generic HandleType)
  // Launcher template: `GPUPipelineLauncher<Pipeline, SR, ValueType, HandleType>`
  // But we have mixed HandleTypes (HandleType2 and HandleType3).
  // The executor handles this via tuples usually, but `GPUPipelineLauncher` takes array pointer!
  // `const HandleType* root_handles`. usage: `root_handles[i]`.
  // This implies ALL handles must be of the same type `HandleType`.
  // This is a limitation of simple `GPUPipelineLauncher` used in tests?
  // In `test_gpu_pipeline_executor.cpp`, everything is `IndexType` (2 columns).
  // Here we have mixed arity.
  // How does the main engine handle mixed arity?
  // `RootExecutor` treats handles as `HandleType`.
  // `NodeHandle` is typically `DeviceSortedArrayIndex::NodeHandle`.
  // Is `NodeHandle` type dependent on arity?
  // `DeviceSortedArrayIndex` template args: `..., KeyTypes...`.
  // `NodeHandle` depends on `DeviceSortedArrayIndex`.
  // If `KeyTypes` diff, `NodeHandle` diff.

  // Issue: `Launcher::execute` takes `HandleType*`.
  // Solution: Cast to a common base / erased type?
  // Or, `NodeHandle` might be generic enough?
  // `NodeHandle` contains `uint32_t*` pointers. It IS generic if `NodeHandle` struct doesn't depend
  // on arity. Let's check `DeviceSortedArrayIndex::NodeHandle`. It is defined in
  // `device_sorted_array_index.h`.

  // If it is generic, we can perform a cast (or reinterpret_cast if layout identical).
  // `NodeHandle` usually stores pointers to columns.

  // Let's check `DeviceSortedArrayIndex` definition.
  // It seems `NodeHandle` is defined inside the class, so it likely depends on template params.

  // If `NodeHandle` is different, we cannot use `GPUPipelineLauncher` as is with an array.
  // We need `GPUPipelineLauncher` to accept a tuple of handles?
  // `GPUPipelineLauncher::execute` signature: `const HandleType* root_handles`.

  // How does `runtime_tc_instantiation.cpp` work?
  // It uses `GPURootExecutor`.
  // `GPURootExecutor`'s `kernel_count` takes `HandleType*`.

  // If all relations in the pipeline don't share the same `HandleType`, we have a problem with this
  // Launcher. Check `test_gpu_pipeline_executor.cpp`: all indices use `IndexType` (2 cols).

  // For my test:
  // R is 2 cols. Z is 3 cols.
  // If `NodeHandle` depends on arity, we can't put them in one array.

  // However, `NodeHandle` for `DeviceSortedArrayIndex` is:
  // struct NodeHandle { const uint32_t* col_ptrs[Arity]; ... } ?
  // If so, size differs.

  // To verify `Let` and `If` on GPU with existing Launcher, I should ensure all relations have SAME
  // Arity/Type? `LetClause` adds a variable. `R(x,y)` (2 vars) -> `Let s=x+y` -> `Z(x,y,s)` (3
  // vars). This inherently changes arity.

  // Does `DeviceSortedArrayIndex` allow `NodeHandle` to be cast?
  // Or maybe I just use a 3-column index for R as well, forcing dummy column?
  // `R(x,y, dummy)`.
  // `IndexSpec` for R can still select columns 0,1.

  // Let's try making EVERYTHING 3-column indices (pad with 0).
  // This ensures `HandleType` is consistent.

  using IndexTypeCommon = IndexType3;
  using HandleType = IndexTypeCommon::NodeHandle;

  IndexTypeCommon index_r_x_c;
  // Pad R to 3 cols: (1,2,0), (3,4,0)...
  build_index_2(reinterpret_cast<IndexType2&>(index_r_x_c), {{1, 2}, {3, 4}, {10, 20}}, {0, 1});
  // Wait, `reinterpret_cast` is unsafe if `IndexType3` has different layout (cols array size).
  // I should use `build_index_3` for R with dummy zeros.

  std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> r_dat = {{1, 2, 0}, {3, 4, 0}, {10, 20, 0}};

  // Helper to build 3-ary index
  auto build_3 = [&](IndexType3& idx, auto& data, std::vector<int> spec) {
    std::size_t n = data.size();
    NDDeviceArray<uint32_t, 3> cols(n);
    DeviceArray<semiring_value_t<BooleanSR>> prov(n);
    std::vector<uint32_t> c0(n), c1(n), c2(n);
    for (size_t i = 0; i < n; ++i) {
      c0[i] = std::get<0>(data[i]);
      c1[i] = std::get<1>(data[i]);
      c2[i] = std::get<2>(data[i]);
    }
    GPU_MEMCPY(cols.column_ptr(0), c0.data(), n * 4, GPU_HOST_TO_DEVICE);
    GPU_MEMCPY(cols.column_ptr(1), c1.data(), n * 4, GPU_HOST_TO_DEVICE);
    GPU_MEMCPY(cols.column_ptr(2), c2.data(), n * 4, GPU_HOST_TO_DEVICE);
    thrust::fill(thrust::device, prov.begin(), prov.end(), BooleanSR::one());
    IndexSpec s;
    s.cols = spec;
    idx.build_from_encoded_device(s, cols, prov);
  };

  IndexTypeCommon index_r_x_3;
  build_3(index_r_x_3, r_dat, {0, 1});  // Using {0,1} spec on 3-col data?
                                        // `build_from_encoded_device` takes `NDDeviceArray`.
                                        // Pass 3 col array.
                                        // Spec {0,1} means valid cols are 0 and 1.

  IndexTypeCommon index_r_xy_3;
  build_3(index_r_xy_3, r_dat, {0, 1});

  IndexTypeCommon index_z_3;
  build_index_3_empty(index_z_3, {0, 1, 2});

  // Now handles are compatible
  auto h_r_x = index_r_x_3.root();
  auto h_r_xy = index_r_xy_3.root();
  auto h_z = index_z_3.root();

  HandleType input_handles[2] = {h_r_x, h_r_xy};
  HandleType output_handles[1] = {h_z};

  using ViewType = typename HandleType::View;
  ViewType input_views[2] = {index_r_x_3.view(), index_r_xy_3.view()};
  ViewType output_views[1] = {index_z_3.view()};

  DeviceArray<HandleType> d_in(2);
  DeviceArray<HandleType> d_out(1);
  DeviceArray<ViewType> d_in_views(2);
  DeviceArray<ViewType> d_out_views(1);
  GPU_MEMCPY(d_in.data(), input_handles, 2 * sizeof(HandleType), GPU_HOST_TO_DEVICE);
  GPU_MEMCPY(d_out.data(), output_handles, 1 * sizeof(HandleType), GPU_HOST_TO_DEVICE);
  GPU_MEMCPY(d_in_views.data(), input_views, 2 * sizeof(ViewType), GPU_HOST_TO_DEVICE);
  GPU_MEMCPY(d_out_views.data(), output_views, 1 * sizeof(ViewType), GPU_HOST_TO_DEVICE);

  // Output buffers
  std::size_t max_out = 100;
  DeviceArray<uint32_t> out_data(max_out * 3);  // Z has 3 cols
  DeviceArray<semiring_value_t<SR>> out_prov(max_out);

  using Launcher = GPUPipelineLauncher<MyPipeline, SR, uint32_t, HandleType>;
  // Get root_unique_values from first source index for partitioning
  uint32_t num_root_keys = h_r_x.degree();
  uint32_t num_unique_root_keys = static_cast<uint32_t>(index_r_x_3.num_unique_root_values());
  // Handle empty/uninitialized index case
  const uint32_t* root_unique_values_ptr =
      (num_unique_root_keys > 0) ? index_r_x_3.root_unique_values().data() : nullptr;

  std::size_t count = Launcher::execute(
      d_in.data(), d_out.data(), d_in_views.data(), d_out_views.data(), root_unique_values_ptr,
      num_unique_root_keys, num_root_keys, out_data.data(), out_prov.data(), max_out);

  device_sync();
  BOOST_CHECK_EQUAL(count, 3);

  // Verify data
  std::vector<uint32_t> host_out(count * 3);
  // Launcher layout: output_stride used?
  // `execute` param: `output_stride`. Passed `max_out`.
  // Layout is column-major.
  // Col 0: out_data [0, max_out)
  // Col 1: out_data [max_out, 2*max_out)
  // Col 2: ...

  std::vector<uint32_t> c0(count), c1(count), c2(count);
  GPU_MEMCPY(c0.data(), out_data.data(), count * 4, GPU_DEVICE_TO_HOST);
  GPU_MEMCPY(c1.data(), out_data.data() + max_out, count * 4, GPU_DEVICE_TO_HOST);
  GPU_MEMCPY(c2.data(), out_data.data() + 2 * max_out, count * 4, GPU_DEVICE_TO_HOST);

  std::set<std::tuple<uint32_t, uint32_t, uint32_t>> results;
  for (size_t i = 0; i < count; ++i)
    results.insert({c0[i], c1[i], c2[i]});

  BOOST_CHECK(results.count({1, 2, 3}));
  BOOST_CHECK(results.count({3, 4, 7}));
  BOOST_CHECK(results.count({10, 20, 30}));
}

BOOST_AUTO_TEST_CASE(test_gpu_if_clause) {
  std::cout << "=== Starting test_gpu_if_clause ===" << std::endl;
  using SR = BooleanSR;

  // R(x) -> Z(x). Both 1 column.
  // We can use IndexType1 (1 col).
  using IndexTypeCommon = IndexType1;
  using HandleType = IndexTypeCommon::NodeHandle;

  using RelationR = AST::RelationSchema<decltype("R"_s), SR, std::tuple<int>>;
  using RelationZ = AST::RelationSchema<decltype("Z"_s), SR, std::tuple<int>>;
  using x_ = AST::Var<decltype("x"_s)>;

  // R(x)
  using SpecR_x = mir::IndexSpecT<RelationR, std::integer_sequence<int, 0>, FULL_VER>;
  using SourceR_x = mir::ColumnSource<SpecR_x, std::tuple<>>;

  // Z(x)
  using IndexZ = mir::IndexSpecT<RelationZ, std::integer_sequence<int, 0>, FULL_VER>;
  using DestZ = mir::DestinationRelation<RelationZ, std::tuple<x_>, NEW_VER, IndexZ>;

  // If x % 2 == 0
  using IfEven =
      AST::IfClause<AST::CppExpr<std::tuple<x_>, [](int x) -> bool { return (x % 2) == 0; }>>;

  using JoinX = mir::ColumnJoin<x_, std::tuple<SourceR_x>, mir::DefaultJoinStrategy, 0,
                                decltype(hana::make_map())>;
  using AllOps = std::tuple<JoinX, IfEven, DestZ>;

  using VarPosMap = decltype(hana::make_map(hana::make_pair(hana::type_c<x_>, hana::int_c<0>)));
  using IteratorPosMap =
      decltype(hana::make_map(hana::make_pair(hana::type_c<SpecR_x>, hana::int_c<0>)));
  using OutputFullIteratorPosMap =
      decltype(hana::make_map(hana::make_pair(hana::type_c<IndexZ>, hana::int_c<0>)));

  struct MyPipeline {
    using mir_ops_type = AllOps;
    using var_pos_map_type = VarPosMap;
    using iterator_pos_map_type = IteratorPosMap;
    using output_full_iterator_pos_map_type = OutputFullIteratorPosMap;
    using destination_specs_type = std::tuple<IndexZ>;
    using all_schemas_type = std::tuple<>;  // Phase 2 requirement
  };

  // Data
  std::vector<uint32_t> r_dat = {1, 2, 3, 4};
  IndexTypeCommon index_r;
  build_index_1(index_r, r_dat, {0});  // Spec {0}

  IndexTypeCommon index_z;
  build_index_1(index_z, {}, {0});

  auto h_r = index_r.root();
  auto h_z = index_z.root();

  HandleType input_handles[1] = {h_r};
  HandleType output_handles[1] = {h_z};

  using ViewType = typename HandleType::View;
  ViewType input_views[1] = {index_r.view()};
  ViewType output_views[1] = {index_z.view()};

  DeviceArray<HandleType> d_in(1), d_out(1);
  DeviceArray<ViewType> d_in_views(1), d_out_views(1);
  GPU_MEMCPY(d_in.data(), input_handles, 1 * sizeof(HandleType), GPU_HOST_TO_DEVICE);
  GPU_MEMCPY(d_out.data(), output_handles, 1 * sizeof(HandleType), GPU_HOST_TO_DEVICE);
  GPU_MEMCPY(d_in_views.data(), input_views, 1 * sizeof(ViewType), GPU_HOST_TO_DEVICE);
  GPU_MEMCPY(d_out_views.data(), output_views, 1 * sizeof(ViewType), GPU_HOST_TO_DEVICE);

  std::size_t max_out = 100;
  DeviceArray<uint32_t> out_data(max_out);
  DeviceArray<semiring_value_t<SR>> out_prov(max_out);

  using Launcher = GPUPipelineLauncher<MyPipeline, SR, uint32_t, HandleType>;
  // Get root_unique_values from first source index for partitioning
  uint32_t num_root_keys = h_r.degree();
  uint32_t num_unique_root_keys = static_cast<uint32_t>(index_r.num_unique_root_values());
  // Handle empty/uninitialized index case
  const uint32_t* root_unique_values_ptr =
      (num_unique_root_keys > 0) ? index_r.root_unique_values().data() : nullptr;

  std::size_t count = Launcher::execute(
      d_in.data(), d_out.data(), d_in_views.data(), d_out_views.data(), root_unique_values_ptr,
      num_unique_root_keys, num_root_keys, out_data.data(), out_prov.data(), max_out);

  device_sync();
  BOOST_CHECK_EQUAL(count, 2);

  std::vector<uint32_t> host_out(count);
  GPU_MEMCPY(host_out.data(), out_data.data(), count * 4, GPU_DEVICE_TO_HOST);

  std::set<uint32_t> res(host_out.begin(), host_out.end());
  BOOST_CHECK(res.count(2));
  BOOST_CHECK(res.count(4));
  BOOST_CHECK(res.count(1) == 0);
  BOOST_CHECK(res.count(3) == 0);
}
