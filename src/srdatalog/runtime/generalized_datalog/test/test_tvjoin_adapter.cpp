#include "column.h"
#include "gpu/device_sorted_array_index.h"
#include "gpu/runtime/adapter/tvjoin.h"
#include <iostream>

using namespace SRDatalog;
using namespace SRDatalog::GPU;

// Mock IndexSpec for testing
struct TestSchema {
  static constexpr std::size_t arity = 2;
  using tuple_type = std::tuple<uint32_t, uint32_t>;
};

struct TestIndexSpec {
  using schema_type = TestSchema;
  using column_indexes_type = std::integer_sequence<int, 0, 1>;
  static constexpr std::size_t size = 2;  // unused but required by concept
};

int main() {
  std::cout << "Verifying TVJoin Adapter Compilation..." << std::endl;

  // Instantiate the template to trigger full compilation of methods/kernels
  using AdapterType = Adapter::SRDatalogToTVJoinISA<TestIndexSpec, uint32_t>;

  // We don't need to run it, just compiling ensures dependencies (RMM, Cuco, TVJoin) are found
  // and kernels are syntactically correct.

  using ReverseAdapterType = Adapter::TVJoinToSRDatalogISA<TestIndexSpec, uint32_t>;

  std::cout << "Adapter templates instantiated successfully." << std::endl;
  return 0;
}
