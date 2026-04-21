#include "andersen_def.h"
#include "gpu/gpu_api.h"  // GPU API abstraction
#include "gpu/init.h"
#include "gpu/runtime/gpu_fixpoint_executor.h"
#include "gpu/runtime/query.h"
#include "mir_printer.h"
#include "runtime/io.h"
#include "test_util.h"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <vector>

using namespace SRDatalog;
using namespace SRDatalog::AST;

using sd::TestUtil::find_project_root;
using Arena = boost::container::pmr::monotonic_buffer_resource;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << (argv[0] ? argv[0] : "andersen_device_benchmark") << " <dataset_dir>"
              << std::endl;
    return 1;
  }

  std::string dataset_path_str = argv[1];
  std::filesystem::path dataset_path(dataset_path_str);

  if (!std::filesystem::exists(dataset_path)) {
    // Try relative to project root if not found
    std::filesystem::path project_root = find_project_root();
    dataset_path = project_root / dataset_path;
  }

  if (!std::filesystem::exists(dataset_path)) {
    std::cerr << "Dataset directory not found: " << dataset_path << std::endl;
    return 1;
  }

  std::filesystem::path addressof_file = dataset_path / "addressOf.csv";
  std::filesystem::path assign_file = dataset_path / "assign.csv";
  std::filesystem::path load_file = dataset_path / "load.csv";
  std::filesystem::path store_file = dataset_path / "store.csv";

  // Check if files exist
  if (!std::filesystem::exists(addressof_file) || !std::filesystem::exists(assign_file) ||
      !std::filesystem::exists(load_file) || !std::filesystem::exists(store_file)) {
    std::cerr << "Dataset files not found in: " << dataset_path << std::endl;
    std::cerr << "Expected: addressOf.csv, assign.csv, load.csv, store.csv" << std::endl;
    return 1;
  }

  sd::GPU::init_cuda();

  // Load EDB
  SemiNaiveDatabase<AndersenBlueprint> source_db;
  std::cout << "Loading data..." << std::endl;
  auto t_load_start = std::chrono::high_resolution_clock::now();

  auto t0 = std::chrono::high_resolution_clock::now();
  load_from_file<AddressOfSchema>(source_db, addressof_file.string());
  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "  AddressOf loaded in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms"
            << std::endl;

  t0 = std::chrono::high_resolution_clock::now();
  load_from_file<AssignSchema>(source_db, assign_file.string());
  t1 = std::chrono::high_resolution_clock::now();
  std::cout << "  Assign loaded in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms"
            << std::endl;

  t0 = std::chrono::high_resolution_clock::now();
  load_from_file<LoadSchema>(source_db, load_file.string());
  t1 = std::chrono::high_resolution_clock::now();
  std::cout << "  Load loaded in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms"
            << std::endl;

  t0 = std::chrono::high_resolution_clock::now();
  load_from_file<StoreSchema>(source_db, store_file.string());
  t1 = std::chrono::high_resolution_clock::now();
  std::cout << "  Store loaded in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms"
            << std::endl;

  auto t_load_end = std::chrono::high_resolution_clock::now();
  std::cout
      << "Total Host Load Time: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t_load_end - t_load_start).count()
      << " ms" << std::endl;

  std::cout << "Initializing Device DB (Host -> Device Copy)..." << std::endl;
  auto t_copy_start = std::chrono::high_resolution_clock::now();
  // Prepare (Host to Device copy)
  auto device_db = SRDatalog::GPU::copy_host_to_device(source_db);
  auto t_copy_end = std::chrono::high_resolution_clock::now();
  std::cout
      << "Host -> Device Copy Time: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t_copy_end - t_copy_start).count()
      << " ms" << std::endl;

  std::cout << "Executing Andersen Analysis..." << std::endl;
  std::cout << "PointsTo relation size (pre-execution): "
            << get_relation_by_schema<PointsToSchema, FULL_VER>(source_db).size() << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  // Execute Base Rule
  SRDatalog::GPU::execute_gpu_query<AndersenBase>(device_db);

  SRDatalog::GPU::inspect<SRDatalog::GPU::GPUQueryExecutor<AndersenFixpoint>::Plan>();
  // Execute Fixpoint
  SRDatalog::GPU::execute_gpu_query<AndersenFixpoint>(device_db);

  GPU_DEVICE_SYNCHRONIZE();
  auto end = std::chrono::high_resolution_clock::now();

  double duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "Finished in " << duration_ms << " ms." << std::endl;
  std::cout << "Andersen MIR: " << std::endl;
  using Plan = typename SRDatalog::GPU::GPUQueryExecutor<AndersenFixpoint>::Plan;
  print_mir<Plan>(std::cout);

  // Reflect back to host to check size (execute_gpu_query might do reflect, but let's be sure
  // or check device index) Standard execute_gpu_query(DeviceDB) executes kernel. Does it
  // reflect? Using explicit reflect check if needed, but for now just measuring kernel time +
  // basic output.

  return 0;
}
