/**
 * @file runtime_tc_tvjoin_instantiation.cpp
 * @brief TVJoin-backed MIR runtime test for Transitive Closure
 *
 * This is a variant of runtime_tc_instantiation.cpp that uses TVJoinMarker
 * to specify DeviceTVJoinIndex as the device index type.
 */

#include <chrono>
#include <iostream>
#include <vector>

#include "../ast.h"
#include "../gpu/device_tvjoin_index.h"
#include "../gpu/gpu_api.h"
#include "../gpu/init.h"
#include "../gpu/runtime/gpu_fixpoint_executor.h"
#include "../gpu/runtime/query.h"
#include "../mir.h"
#include "../relation_col.h"
#include "../runtime/executor.h"
#include "../runtime/io.h"
#include "../runtime/state.h"
#include "../semiring.h"

namespace sd = SRDatalog;
namespace ast = sd::AST;
namespace mir = sd::mir;

using ast::database;
using ast::relation;
using ast::operator""_v;
using ast::Literals::operator""_s;

namespace {

using SR = BooleanSR;

// Variables
constexpr auto x = "x"_v;
constexpr auto y = "y"_v;
constexpr auto z = "z"_v;

using VarX = decltype(x)::type;
using VarY = decltype(y)::type;
using VarZ = decltype(z)::type;

// Schema definitions using TVJoinMarker for TVJoin device index
// Use RelationSchema directly (not the `relation` alias) to specify index type
using PathSchema =
    ast::RelationSchema<decltype("Path"_s), SR, std::tuple<int, int>, sd::GPU::TVJoinMarker>;
using EdgeSchema =
    ast::RelationSchema<decltype("Edge"_s), SR, std::tuple<int, int>, sd::GPU::TVJoinMarker>;

// Define IndexSpecs
using IndexEdge_YZ = sd::mir::IndexSpecT<EdgeSchema, std::integer_sequence<int, 0, 1>, FULL_VER>;
using IndexPath_YX = sd::mir::IndexSpecT<PathSchema, std::integer_sequence<int, 1, 0>, DELTA_VER>;
using IndexPath_Full_YX =
    sd::mir::IndexSpecT<PathSchema, std::integer_sequence<int, 1, 0>, FULL_VER>;
using IndexPath_New_YX = sd::mir::IndexSpecT<PathSchema, std::integer_sequence<int, 1, 0>, NEW_VER>;

// Define Pipeline
using SourcePath = mir::ColumnSource<IndexPath_YX, std::tuple<>>;
using SourceEdge = mir::ColumnSource<IndexEdge_YZ, std::tuple<>>;

using JoinY_T =
    mir::ColumnJoin<VarY, std::tuple<SourcePath, SourceEdge>, mir::DefaultJoinStrategy, 0, void>;

using SourcePath_X = mir::ColumnSource<IndexPath_YX, std::tuple<VarY>>;
using SourceEdge_Z = mir::ColumnSource<IndexEdge_YZ, std::tuple<VarY>>;

using JoinXZ_T =
    mir::CartesianJoin<std::tuple<VarX, VarZ>, std::tuple<SourcePath_X, SourceEdge_Z>, 0, void>;

using DestOut_T =
    mir::DestinationRelation<PathSchema, std::tuple<VarX, VarZ>, NEW_VER, IndexPath_Full_YX>;

using MIROps = std::tuple<JoinY_T, JoinXZ_T, DestOut_T>;
using JoinPlan = std::tuple<VarY, VarX, VarZ>;
using VarPosMap = typename mir::ComputeVarPosMap<JoinPlan>::type;

using TCPipeline = mir::Pipeline<MIROps, VarPosMap>;

// Define Instructions - using ComputeDelta for proper deduplication
using Instructions =
    std::tuple<mir::ExecutePipeline<TCPipeline>, mir::ClearRelation<PathSchema, DELTA_VER>,
               mir::CheckSize<PathSchema, NEW_VER>, mir::RebuildIndex<IndexPath_New_YX>,
               mir::ComputeDelta<PathSchema, NEW_VER, FULL_VER, DELTA_VER, IndexPath_Full_YX>,
               mir::MergeIndex<IndexPath_Full_YX>, mir::ClearRelation<PathSchema, NEW_VER>>;

// Schema types
using SchemaTuple = database<EdgeSchema, PathSchema>;

// Use TVJoin-backed device relation type via the trait system
using DeviceDB = ast::SemiNaiveDatabase<SchemaTuple, sd::GPU::DeviceRelationType>;

using FixpointExecutor = sd::GPU::GPUFixpointExecutor<Instructions, DeviceDB>;

}  // namespace

int main(int argc, char** argv) {
  sd::GPU::init_cuda();

  // 1. Setup Host DB (uses HashTrieIndex for loading)
  using HostRelation = sd::GPU::HostRelationType<EdgeSchema>;
  ast::SemiNaiveDatabase<SchemaTuple, sd::GPU::HostRelationType> host_db;

  // Load edges
  std::string edge_path = "test_data/Edge.csv";
  if (argc > 1) {
    edge_path = argv[1];
  }

  auto& edge_full_host = std::get<0>(host_db.full);
  edge_full_host.set_version(FULL_VER);
  sd::load_file(edge_full_host, edge_path);
  std::cout << "Loaded " << edge_full_host.size() << " edges." << std::endl;

  // Init Path(Full) with Edges
  auto& path_full_host = std::get<1>(host_db.full);
  path_full_host.set_version(FULL_VER);
  path_full_host.concat(edge_full_host);

  // Init Path(Delta) with Edges
  auto& path_delta_host = std::get<1>(host_db.delta);
  path_delta_host.set_version(DELTA_VER);
  path_delta_host.concat(edge_full_host);

  // 2. Prepare Device DB (uses DeviceTVJoinIndex via trait mapping)
  DeviceDB device_db;
  sd::DatabaseInitializer<decltype(host_db), DeviceDB>::execute(host_db, device_db);
  std::cout << "Device DB Initialized with TVJoin indices." << std::endl;

  // 3. Execution Loop
  auto start = std::chrono::high_resolution_clock::now();
  std::cout << "Executing Fixpoint..." << std::endl;

  constexpr size_t MAX_ITERS = 1000;
  auto& path_delta_dev = std::get<1>(device_db.delta);
  auto& path_full_dev = std::get<1>(device_db.full);

  for (size_t i = 0; i < MAX_ITERS; ++i) {
    if (path_delta_dev.interned_size() == 0) {
      std::cout << "Converged at iteration " << i << std::endl;
      break;
    }
    std::cout << "Iteration " << i << ": delta size = " << path_delta_dev.interned_size()
              << ", full size = " << path_full_dev.interned_size() << std::endl;
    FixpointExecutor::execute(device_db, i, MAX_ITERS);
  }

  GPU_DEVICE_SYNCHRONIZE();
  auto end = std::chrono::high_resolution_clock::now();

  std::cout << "Finished in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms."
            << std::endl;

  // 4. Retrieve Results
  auto path_full_host_res = sd::GPU::HostRelationType<PathSchema>::to_host(path_full_dev);
  path_full_host_res.reconstruct_columns_from_interned();

  std::cout << "Total TC size: " << path_full_host_res.size() << std::endl;

  return 0;
}
