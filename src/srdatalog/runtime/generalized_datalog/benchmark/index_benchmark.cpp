// Benchmarks for SortedArrayIndex construction and queries.
// Uses Google Benchmark (package "benchmark" in xmake.lua).
//
// We measure three operations:
//  - build_from_encoded: build index from encoded tuples for N = 1M, 2M, 10M
//  - merge: merge delta relations of sizes 1M, 2M, 10M into a 5M base index
//  - search: prefix lookups on a large relation using pre-generated query keys
//  - comparison: SortedArrayIndex vs HashTrieIndex
// #include <mimalloc-new-delete.h>  // Use mimalloc for all dynamic allocations in this benchmark
//
#include "hashtrie.h"
#include "semiring.h"
#include "sorted_array_index.h"
#include "system.h"

#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace sd = SRDatalog;
using sd::default_memory_resource;
using sd::HashTrieIndex;
using sd::IndexSpec;
using sd::memory_resource;
using sd::SortedArrayIndex;
using sd::Vector;

using ::BooleanSR;
using SR = BooleanSR;

// -----------------------------------------------------------------------------
// Index configuration per arity
// -----------------------------------------------------------------------------

template <std::size_t Arity>
struct IndexConfig;

template <>
struct IndexConfig<2> {
  using Tuple = std::tuple<int, int>;
  using Index = SortedArrayIndex<SR, Tuple>;
  static inline const IndexSpec spec{{0, 1}};  // index on first two columns
  static constexpr const char* arity_label = "arity2";
};

template <>
struct IndexConfig<3> {
  using Tuple = std::tuple<int, int, int>;
  using Index = SortedArrayIndex<SR, Tuple>;
  static inline const IndexSpec spec{{0, 1}};
  static constexpr const char* arity_label = "arity3";
};

template <>
struct IndexConfig<4> {
  using Tuple = std::tuple<int, int, int, int>;
  using Index = SortedArrayIndex<SR, Tuple>;
  static inline const IndexSpec spec{{0, 1}};
  static constexpr const char* arity_label = "arity4";
};

// -----------------------------------------------------------------------------
// Synthetic Data Generation (templated by arity)
// -----------------------------------------------------------------------------

template <std::size_t Arity>
struct EncodedEnv {
  std::array<Vector<uint32_t>, Arity> cols;
};

// Simple cache for reusing generated encoded tuples at different sizes and arities.
template <std::size_t Arity>
static EncodedEnv<Arity>& get_env(std::size_t n) {
  static std::unordered_map<std::size_t, EncodedEnv<Arity>> cache;

  auto it = cache.find(n);
  if (it != cache.end()) {
    return it->second;
  }

  EncodedEnv<Arity> env;
  for (auto& col : env.cols) {
    col = Vector<uint32_t>(default_memory_resource());
    col.resize(n);
  }

  // Deterministic RNG for reproducibility.
  std::mt19937_64 rng(42 + static_cast<uint64_t>(n) + static_cast<uint64_t>(Arity) * 997u);

  // Column 0: moderately small domain to create duplicates.
  std::uniform_int_distribution<std::size_t> dist_key(0, std::max<std::size_t>(1, n / 32));
  // Column 1: larger spread.
  std::uniform_int_distribution<std::size_t> dist_val(0, n);
  // Remaining columns: arbitrary payload.
  std::uniform_int_distribution<std::size_t> dist_payload(0, n * 2);

  for (std::size_t i = 0; i < n; ++i) {
    env.cols[0][i] = static_cast<uint32_t>(dist_key(rng));
    if constexpr (Arity > 1) {
      env.cols[1][i] = static_cast<uint32_t>(dist_val(rng));
    }
    for (std::size_t c = 2; c < Arity; ++c) {
      env.cols[c][i] = static_cast<uint32_t>(dist_payload(rng));
    }
  }

  auto [insert_it, _] = cache.emplace(n, std::move(env));
  return insert_it->second;
}

template <std::size_t Arity>
static std::array<std::span<const uint32_t>, Arity> make_span(const EncodedEnv<Arity>& env) {
  std::array<std::span<const uint32_t>, Arity> spans{};
  for (std::size_t c = 0; c < Arity; ++c) {
    spans[c] = std::span(env.cols[c].data(), env.cols[c].size());
  }
  return spans;
}

// -----------------------------------------------------------------------------
// Build Benchmarks
// -----------------------------------------------------------------------------

template <std::size_t Arity>
static void BM_SAI_Build(benchmark::State& state) {
  using Cfg = IndexConfig<Arity>;
  using Index = typename Cfg::Index;

  std::size_t n = static_cast<std::size_t>(state.range(0));

  auto& env = get_env<Arity>(n);
  auto spans = make_span<Arity>(env);

  for (auto _ : state) {
    Index idx(default_memory_resource());
    idx.build_from_encoded(Cfg::spec, spans, default_memory_resource());

    benchmark::DoNotOptimize(idx.root());
    // Report index memory usage.
    state.counters["bytes"] = static_cast<double>(idx.bytes_used());
  }

  state.SetLabel(std::string("build_from_encoded/") + Cfg::arity_label);
}

// -----------------------------------------------------------------------------
// Merge Benchmarks
// -----------------------------------------------------------------------------

template <std::size_t Arity>
static void BM_SAI_MergeInto5M(benchmark::State& state) {
  using Cfg = IndexConfig<Arity>;
  using Index = typename Cfg::Index;

  constexpr std::size_t kBaseSize = 5'000'000;
  std::size_t delta_n = static_cast<std::size_t>(state.range(0));

  auto& base_env = get_env<Arity>(kBaseSize);
  auto& delta_env = get_env<Arity>(delta_n);

  auto base_spans = make_span<Arity>(base_env);
  auto delta_spans = make_span<Arity>(delta_env);

  for (auto _ : state) {
    state.PauseTiming();
    // Rebuild base and delta indexes for each iteration to keep merge timing clean.
    Index base_idx(default_memory_resource());
    Index delta_idx(default_memory_resource());
    base_idx.build_from_encoded(Cfg::spec, base_spans, default_memory_resource());
    delta_idx.build_from_encoded(Cfg::spec, delta_spans, default_memory_resource());
    state.ResumeTiming();

    // Merge delta into base. Row IDs from delta are offset by base size.
    base_idx.merge(delta_idx, kBaseSize);

    benchmark::DoNotOptimize(base_idx.root());

    state.PauseTiming();
    state.counters["bytes"] = static_cast<double>(base_idx.bytes_used());
    state.ResumeTiming();
  }

  state.SetLabel(std::string("merge_into_5M/") + Cfg::arity_label);
}

// -----------------------------------------------------------------------------
// Search Benchmarks
// -----------------------------------------------------------------------------

template <std::size_t Arity>
static void BM_SAI_Search(benchmark::State& state) {
  using Cfg = IndexConfig<Arity>;
  using Index = typename Cfg::Index;

  // Use a large enough relation for search.
  constexpr std::size_t kSearchSize = 10'000'000;
  constexpr std::size_t kNumQueries = 100'000;

  auto& env = get_env<Arity>(kSearchSize);
  auto spans = make_span<Arity>(env);

  Index idx(default_memory_resource());
  idx.build_from_encoded(Cfg::spec, spans, default_memory_resource());

  // Pre-generate query keys from the same distribution as column 0.
  std::vector<uint32_t> query_keys;
  query_keys.reserve(kNumQueries);

  std::mt19937_64 rng(1337);
  std::uniform_int_distribution<std::size_t> dist_key(0,
                                                      std::max<std::size_t>(1, kSearchSize / 32));

  for (std::size_t i = 0; i < kNumQueries; ++i) {
    query_keys.push_back(static_cast<uint32_t>(dist_key(rng)));
  }

  auto root = idx.root();

  for (auto _ : state) {
    std::size_t total_hits = 0;
    for (uint32_t key : query_keys) {
      auto node = root.prefix(key);
      if (node.valid()) {
        total_hits += node.degree();
      }
    }
    benchmark::DoNotOptimize(total_hits);
  }

  state.counters["bytes"] = static_cast<double>(idx.bytes_used());
  state.SetLabel(std::string("prefix_search/") + Cfg::arity_label);
}

// -----------------------------------------------------------------------------
// Comparison Benchmarks: SAI vs HashTrieIndex
// -----------------------------------------------------------------------------

// Note: With uint32_t as the default ValueType for both SortedArrayIndex and HashTrieIndex,
// we can use the same uint32_t spans directly for both index types.

template <std::size_t Arity>
static void BM_Compare_Build_SAI(benchmark::State& state) {
  using Cfg = IndexConfig<Arity>;
  using Index = typename Cfg::Index;

  std::size_t n = static_cast<std::size_t>(state.range(0));
  auto& env = get_env<Arity>(n);
  auto spans = make_span<Arity>(env);

  for (auto _ : state) {
    Index idx(default_memory_resource());
    idx.build_from_encoded(Cfg::spec, spans, default_memory_resource());
    benchmark::DoNotOptimize(idx.root());
    state.counters["bytes"] = static_cast<double>(idx.bytes_used());
  }

  state.SetLabel(std::string("SAI/build/") + Cfg::arity_label);
}

template <std::size_t Arity>
static void BM_Compare_Build_HashTrie(benchmark::State& state) {
  using Cfg = IndexConfig<Arity>;
  using HashTrie = HashTrieIndex<SR, typename Cfg::Tuple>;

  std::size_t n = static_cast<std::size_t>(state.range(0));
  auto& env = get_env<Arity>(n);
  auto spans = make_span<Arity>(env);

  for (auto _ : state) {
    HashTrie idx(default_memory_resource());
    idx.build_from_encoded(Cfg::spec, spans, default_memory_resource());
    benchmark::DoNotOptimize(idx.root());
    state.counters["bytes"] = static_cast<double>(idx.bytes_used());
  }

  state.SetLabel(std::string("HashTrie/build/") + Cfg::arity_label);
}

template <std::size_t Arity>
static void BM_Compare_Merge_HashTrie(benchmark::State& state) {
  using Cfg = IndexConfig<Arity>;
  using HashTrie = HashTrieIndex<SR, typename Cfg::Tuple>;

  constexpr std::size_t kBaseSize = 5'000'000;
  std::size_t delta_n = static_cast<std::size_t>(state.range(0));

  auto& base_env = get_env<Arity>(kBaseSize);
  auto& delta_env = get_env<Arity>(delta_n);

  auto base_spans = make_span<Arity>(base_env);
  auto delta_spans = make_span<Arity>(delta_env);

  for (auto _ : state) {
    state.PauseTiming();
    // Rebuild base and delta indexes for each iteration to keep merge timing clean.
    HashTrie base_idx(default_memory_resource());
    HashTrie delta_idx(default_memory_resource());

    base_idx.build_from_encoded(Cfg::spec, base_spans, default_memory_resource());
    delta_idx.build_from_encoded(Cfg::spec, delta_spans, default_memory_resource());
    state.ResumeTiming();

    // Merge delta into base. Row IDs from delta are offset by base size.
    base_idx.merge(delta_idx, static_cast<uint32_t>(kBaseSize));

    benchmark::DoNotOptimize(base_idx.root());

    state.PauseTiming();
    state.counters["bytes"] = static_cast<double>(base_idx.bytes_used());
    state.ResumeTiming();
  }

  state.SetLabel(std::string("HashTrie/merge_into_5M/") + Cfg::arity_label);
}

template <std::size_t Arity>
static void BM_Compare_Search_SAI(benchmark::State& state) {
  using Cfg = IndexConfig<Arity>;
  using Index = typename Cfg::Index;

  constexpr std::size_t kSearchSize = 10'000'000;
  constexpr std::size_t kNumQueries = 100'000;

  auto& env = get_env<Arity>(kSearchSize);
  auto spans = make_span<Arity>(env);

  Index idx(default_memory_resource());
  idx.build_from_encoded(Cfg::spec, spans, default_memory_resource());

  std::vector<uint32_t> query_keys;
  query_keys.reserve(kNumQueries);
  std::mt19937_64 rng(1337);
  std::uniform_int_distribution<std::size_t> dist_key(0,
                                                      std::max<std::size_t>(1, kSearchSize / 32));
  for (std::size_t i = 0; i < kNumQueries; ++i) {
    query_keys.push_back(static_cast<uint32_t>(dist_key(rng)));
  }

  auto root = idx.root();

  for (auto _ : state) {
    std::size_t total_hits = 0;
    for (uint32_t key : query_keys) {
      auto node = root.prefix(key);
      if (node.valid()) {
        total_hits += node.degree();
      }
    }
    benchmark::DoNotOptimize(total_hits);
  }

  state.counters["bytes"] = static_cast<double>(idx.bytes_used());
  state.SetLabel(std::string("SAI/search/") + Cfg::arity_label);
}

template <std::size_t Arity>
static void BM_Compare_Search_HashTrie(benchmark::State& state) {
  using Cfg = IndexConfig<Arity>;
  using HashTrie = HashTrieIndex<SR, typename Cfg::Tuple>;

  constexpr std::size_t kSearchSize = 10'000'000;
  constexpr std::size_t kNumQueries = 100'000;

  auto& env = get_env<Arity>(kSearchSize);
  auto spans = make_span<Arity>(env);

  HashTrie idx(default_memory_resource());
  idx.build_from_encoded(Cfg::spec, spans, default_memory_resource());

  std::vector<uint32_t> query_keys;
  query_keys.reserve(kNumQueries);
  std::mt19937_64 rng(1337);
  std::uniform_int_distribution<std::size_t> dist_key(0,
                                                      std::max<std::size_t>(1, kSearchSize / 32));
  for (std::size_t i = 0; i < kNumQueries; ++i) {
    query_keys.push_back(static_cast<uint32_t>(dist_key(rng)));
  }

  auto root = idx.root();

  for (auto _ : state) {
    std::size_t total_hits = 0;
    for (uint32_t key : query_keys) {
      auto node = root.prefix(key);
      if (node.valid()) {
        total_hits += node.degree();
      }
    }
    benchmark::DoNotOptimize(total_hits);
  }

  state.counters["bytes"] = static_cast<double>(idx.bytes_used());
  state.SetLabel(std::string("HashTrie/search/") + Cfg::arity_label);
}

// -----------------------------------------------------------------------------
// Benchmark Registration
// -----------------------------------------------------------------------------
// Organized for easy comparison: SAI and HashTrie are grouped together
// by operation type and arity

// ============================================================================
// Build Benchmarks - Arity 2
// ============================================================================
BENCHMARK_TEMPLATE(BM_Compare_Build_SAI, 2)
    ->Arg(1'000'000)
    ->Arg(2'000'000)
    ->Arg(10'000'000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(BM_Compare_Build_HashTrie, 2)
    ->Arg(1'000'000)
    ->Arg(2'000'000)
    ->Arg(10'000'000)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Build Benchmarks - Arity 3
// ============================================================================
BENCHMARK_TEMPLATE(BM_Compare_Build_SAI, 3)
    ->Arg(1'000'000)
    ->Arg(2'000'000)
    ->Arg(10'000'000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(BM_Compare_Build_HashTrie, 3)
    ->Arg(1'000'000)
    ->Arg(2'000'000)
    ->Arg(10'000'000)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Build Benchmarks - Arity 4
// ============================================================================
BENCHMARK_TEMPLATE(BM_SAI_Build, 4)
    ->Arg(1'000'000)
    ->Arg(2'000'000)
    ->Arg(10'000'000)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Search Benchmarks - Arity 2
// ============================================================================
BENCHMARK_TEMPLATE(BM_Compare_Search_SAI, 2)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Compare_Search_HashTrie, 2)->Unit(benchmark::kMillisecond);

// ============================================================================
// Search Benchmarks - Arity 3
// ============================================================================
BENCHMARK_TEMPLATE(BM_Compare_Search_SAI, 3)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Compare_Search_HashTrie, 3)->Unit(benchmark::kMillisecond);

// ============================================================================
// Search Benchmarks - Arity 4
// ============================================================================
BENCHMARK_TEMPLATE(BM_SAI_Search, 4)->Unit(benchmark::kMillisecond);

// ============================================================================
// Merge Benchmarks - Arity 2
// ============================================================================
BENCHMARK_TEMPLATE(BM_SAI_MergeInto5M, 2)
    ->Arg(1'000'000)
    ->Arg(2'000'000)
    ->Arg(10'000'000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(BM_Compare_Merge_HashTrie, 2)
    ->Arg(1'000'000)
    ->Arg(2'000'000)
    ->Arg(10'000'000)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Merge Benchmarks - Arity 3
// ============================================================================
BENCHMARK_TEMPLATE(BM_SAI_MergeInto5M, 3)
    ->Arg(1'000'000)
    ->Arg(2'000'000)
    ->Arg(10'000'000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(BM_Compare_Merge_HashTrie, 3)
    ->Arg(1'000'000)
    ->Arg(2'000'000)
    ->Arg(10'000'000)
    ->Unit(benchmark::kMillisecond);

// ============================================================================
// Merge Benchmarks - Arity 4
// ============================================================================
BENCHMARK_TEMPLATE(BM_SAI_MergeInto5M, 4)
    ->Arg(1'000'000)
    ->Arg(2'000'000)
    ->Arg(10'000'000)
    ->Unit(benchmark::kMillisecond);

int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}
