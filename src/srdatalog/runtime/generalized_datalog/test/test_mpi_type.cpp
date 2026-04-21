#include <cassert>
#include <iostream>
#include <string>
#include <typeindex>
#include <type_traits>

#include <mpi.h>
#include "mpi_type.h"

// --------- tiny helper: ensure dt has same size as sizeof(T) ----------
template <class T>
static void expect_mpi_size_matches(MPI_Datatype dt, const char* msg) {
  assert(dt != MPI_DATATYPE_NULL && "MPI_Datatype should not be NULL");
  int sz = -1;
  int rc = MPI_Type_size(dt, &sz);
  assert(rc == MPI_SUCCESS && "MPI_Type_size failed");
  if (static_cast<int>(sizeof(T)) != sz) {
    std::cerr << "Size mismatch for " << msg
              << " | sizeof(T)=" << sizeof(T)
              << " MPI_Type_size=" << sz << "\n";
    std::abort();
  }
}

// --------- tests ---------

static void test_traits_static() {
  // Directly mappable: integers/floats/bool/char family/long double/MPI_Aint
  static_assert(is_mpi_direct_mappable_v<std::int8_t>);
  static_assert(is_mpi_direct_mappable_v<std::uint8_t>);
  static_assert(is_mpi_direct_mappable_v<std::int16_t>);
  static_assert(is_mpi_direct_mappable_v<std::uint16_t>);
  static_assert(is_mpi_direct_mappable_v<std::int32_t>);
  static_assert(is_mpi_direct_mappable_v<std::uint32_t>);
  static_assert(is_mpi_direct_mappable_v<std::int64_t>);
  static_assert(is_mpi_direct_mappable_v<std::uint64_t>);
  static_assert(is_mpi_direct_mappable_v<float>);
  static_assert(is_mpi_direct_mappable_v<double>);
  static_assert(is_mpi_direct_mappable_v<long double>);
  static_assert(is_mpi_direct_mappable_v<bool>);
  static_assert(is_mpi_direct_mappable_v<char>);
  static_assert(is_mpi_direct_mappable_v<signed char>);
  static_assert(is_mpi_direct_mappable_v<unsigned char>);
#if !defined(__APPLE__)
  static_assert(is_mpi_direct_mappable_v<MPI_Aint>);
#endif

  // Not directly mappable
  static_assert(!is_mpi_direct_mappable_v<std::string>);
  struct Foo { int x; double y; };
  static_assert(!is_mpi_direct_mappable_v<Foo>);
}

static void test_compile_time_mapper_sizes() {
  // Ensure compile-time mapper returns a type whose size matches sizeof(T)
  expect_mpi_size_matches<std::int8_t>  (mpi_datatype_of<std::int8_t>(),   "int8_t");
  expect_mpi_size_matches<std::uint8_t> (mpi_datatype_of<std::uint8_t>(),  "uint8_t");
  expect_mpi_size_matches<std::int16_t> (mpi_datatype_of<std::int16_t>(),  "int16_t");
  expect_mpi_size_matches<std::uint16_t>(mpi_datatype_of<std::uint16_t>(), "uint16_t");
  expect_mpi_size_matches<std::int32_t> (mpi_datatype_of<std::int32_t>(),  "int32_t");
  expect_mpi_size_matches<std::uint32_t>(mpi_datatype_of<std::uint32_t>(), "uint32_t");
  expect_mpi_size_matches<std::int64_t> (mpi_datatype_of<std::int64_t>(),  "int64_t");
  expect_mpi_size_matches<std::uint64_t>(mpi_datatype_of<std::uint64_t>(), "uint64_t");

  expect_mpi_size_matches<float>        (mpi_datatype_of<float>(),         "float");
  expect_mpi_size_matches<double>       (mpi_datatype_of<double>(),        "double");
  expect_mpi_size_matches<long double>  (mpi_datatype_of<long double>(),   "long double");

  expect_mpi_size_matches<bool>         (mpi_datatype_of<bool>(),          "bool");
  expect_mpi_size_matches<char>         (mpi_datatype_of<char>(),          "char");
  expect_mpi_size_matches<signed char>  (mpi_datatype_of<signed char>(),   "signed char");
  expect_mpi_size_matches<unsigned char>(mpi_datatype_of<unsigned char>(), "unsigned char");

  expect_mpi_size_matches<MPI_Aint>     (mpi_datatype_of<MPI_Aint>(),      "MPI_Aint");
}

static void test_runtime_type_index_mapper() {
  // For direct types: type_index mapping must yield correct sizes
  auto chk = [](auto sample, const char* msg) {
    using T = decltype(sample);
    MPI_Datatype dt = mpi_datatype_from_type_index(std::type_index(typeid(T)));
    expect_mpi_size_matches<T>(dt, msg);
  };

  chk(int32_t{},  "type_index int32_t");
  chk(uint64_t{}, "type_index uint64_t");
  chk(float{},    "type_index float");
  chk(double{},   "type_index double");
  chk(static_cast<long double>(0), "type_index long double");
  chk(bool{},     "type_index bool");
  chk(char{},     "type_index char");
  chk(static_cast<signed char>(0), "type_index signed char");
  chk(static_cast<unsigned char>(0), "type_index unsigned char");
  chk(MPI_Aint{}, "type_index MPI_Aint");

  // std::string should NOT be directly mappable at runtime
  MPI_Datatype str_dt = mpi_datatype_from_type_index(std::type_index(typeid(std::string)));
  assert(str_dt == MPI_DATATYPE_NULL && "std::string must not have a direct MPI_Datatype");
}

static void test_size_based_helpers() {
  // size_t
  {
    MPI_Datatype dt = mpi_datatype_of_size_t();
    assert(dt != MPI_DATATYPE_NULL && "size_t mapping must succeed");
    expect_mpi_size_matches<std::size_t>(dt, "size_t");
  }
  // ptrdiff_t
  {
    MPI_Datatype dt = mpi_datatype_of_ptrdiff_t();
    assert(dt != MPI_DATATYPE_NULL && "ptrdiff_t mapping must succeed");
    expect_mpi_size_matches<std::ptrdiff_t>(dt, "ptrdiff_t");
  }
}

static void test_create_contiguous_datatype() {
  struct POD {
    int a;
    double b;
  };
  static_assert(std::is_trivially_copyable_v<POD>, "POD must be trivially copyable");

  MPI_Datatype dt = create_contiguous_datatype<POD>();
  assert(dt != MPI_DATATYPE_NULL && "Derived MPI_Datatype should be created");

  // Size should match sizeof(POD) (contiguous bytes representation)
  expect_mpi_size_matches<POD>(dt, "POD derived type");

  // Cleanup
  int rc = MPI_Type_free(&dt);
  assert(rc == MPI_SUCCESS && "MPI_Type_free failed");
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  test_traits_static();
  test_compile_time_mapper_sizes();
  test_runtime_type_index_mapper();
  test_size_based_helpers();
  test_create_contiguous_datatype();

  if (int rank = 0; (MPI_Comm_rank(MPI_COMM_WORLD, &rank), true)) {
    if (rank == 0) std::cout << "[OK] mpi_type.h tests passed.\n";
  }

  MPI_Finalize();
  return 0;
}
