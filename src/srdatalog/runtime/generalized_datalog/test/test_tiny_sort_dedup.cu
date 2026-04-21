/**
 * Rigorous benchmark: fused tiny sort+dedup for rebuild_index replacement
 *
 * Tests:
 *   1. Self-dedup correctness (internal duplicates removed)
 *   2. Output is SORTED (not just deduped) — required for downstream set_difference
 *   3. Arity 3 and 4 with SoA multi-column layout
 *   4. Various sizes (5, 20, 100, 500, 1000)
 *   5. End-to-end: tiny_sort → set_difference against FULL → verify delta matches thrust
 *   6. Multi-iteration: simulate N fixpoint iterations accumulating into FULL
 *   7. Sort key correctness: tuples differing only in col2/col3 must sort correctly
 *   8. Performance comparison vs thrust pipeline
 */

#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <cassert>
#include <set>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/set_operations.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

// ============================================================
// SoA helpers
// ============================================================
struct SoARelation {
    uint32_t* d_data = nullptr;
    uint32_t count = 0;
    uint32_t stride = 0;
    int arity = 0;
    void alloc(int n, int ar) {
        arity = ar; stride = n; count = n;
        cudaMalloc(&d_data, arity * stride * sizeof(uint32_t));
    }
    void free() { if (d_data) { cudaFree(d_data); d_data = nullptr; } }
    // Read back to host
    std::vector<std::vector<uint32_t>> to_host() const {
        std::vector<uint32_t> h(arity * stride);
        cudaMemcpy(h.data(), d_data, arity * stride * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        std::vector<std::vector<uint32_t>> tuples(count, std::vector<uint32_t>(arity));
        for (uint32_t i = 0; i < count; i++)
            for (int j = 0; j < arity; j++)
                tuples[i][j] = h[j * stride + i];
        return tuples;
    }
    void from_host(const std::vector<std::vector<uint32_t>>& tuples) {
        uint32_t n = tuples.size();
        int ar = tuples.empty() ? arity : tuples[0].size();
        if (!d_data || n > stride) {
            free();
            alloc(n, ar);
        }
        count = n; arity = ar;
        std::vector<uint32_t> h(arity * n);
        for (uint32_t i = 0; i < n; i++)
            for (int j = 0; j < arity; j++)
                h[j * n + i] = tuples[i][j];
        cudaMemcpy(d_data, h.data(), arity * n * sizeof(uint32_t), cudaMemcpyHostToDevice);
    }
};

// Include the actual kernel from the runtime header
#include "gpu/runtime/jit/tiny_sort_dedup.h"
using namespace SRDatalog::GPU::JIT;

static constexpr int kSortBlock = kTinySortBlock;
static constexpr int kItemsPerThread = kTinySortItemsPerThread;
static constexpr int kMaxItems = kTinySortMaxItems;

// Kernel is included from tiny_sort_dedup.h above

// ============================================================
// CPU reference: sort + unique
// ============================================================
using Tuple = std::vector<uint32_t>;

std::vector<Tuple> cpu_sort_dedup(const std::vector<Tuple>& input) {
    auto tuples = input;
    std::sort(tuples.begin(), tuples.end());
    tuples.erase(std::unique(tuples.begin(), tuples.end()), tuples.end());
    return tuples;
}

std::vector<Tuple> cpu_set_difference(const std::vector<Tuple>& a, const std::vector<Tuple>& b) {
    std::vector<Tuple> result;
    std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(result));
    return result;
}

// ============================================================
// Thrust baseline
// ============================================================
template<int Arity>
struct SoAComparator {
    const uint32_t* data; uint32_t stride;
    __host__ __device__ bool operator()(uint32_t a, uint32_t b) const {
        for (int c = 0; c < Arity; c++) {
            uint32_t va = data[c * stride + a], vb = data[c * stride + b];
            if (va < vb) return true;
            if (va > vb) return false;
        }
        return false;
    }
};

template<int Arity>
struct SoAEqual {
    const uint32_t* data; uint32_t stride;
    __host__ __device__ bool operator()(uint32_t a, uint32_t b) const {
        for (int c = 0; c < Arity; c++)
            if (data[c * stride + a] != data[c * stride + b]) return false;
        return true;
    }
};

template<int Arity>
std::vector<Tuple> run_thrust_sort_dedup(SoARelation& in) {
    uint32_t N = in.count;
    thrust::device_vector<uint32_t> perm(N);
    thrust::sequence(perm.begin(), perm.end());
    SoAComparator<Arity> cmp{in.d_data, in.stride};
    thrust::sort(perm.begin(), perm.end(), cmp);

    thrust::device_vector<uint32_t> sorted(Arity * N);
    for (int c = 0; c < Arity; c++)
        thrust::gather(perm.begin(), perm.end(),
                       thrust::device_pointer_cast(in.d_data + c * in.stride),
                       sorted.begin() + c * N);

    thrust::device_vector<uint32_t> uniq_perm(N);
    thrust::sequence(uniq_perm.begin(), uniq_perm.end());
    SoAEqual<Arity> eq{thrust::raw_pointer_cast(sorted.data()), N};
    auto end = thrust::unique(uniq_perm.begin(), uniq_perm.end(), eq);
    uint32_t uniq_count = end - uniq_perm.begin();

    // Read back sorted+deduped tuples
    std::vector<uint32_t> h_sorted(Arity * N);
    cudaMemcpy(h_sorted.data(), thrust::raw_pointer_cast(sorted.data()),
               Arity * N * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    std::vector<uint32_t> h_perm(uniq_count);
    cudaMemcpy(h_perm.data(), thrust::raw_pointer_cast(uniq_perm.data()),
               uniq_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::vector<Tuple> result(uniq_count, Tuple(Arity));
    for (uint32_t i = 0; i < uniq_count; i++)
        for (int c = 0; c < Arity; c++)
            result[i][c] = h_sorted[c * N + h_perm[i]];
    cudaDeviceSynchronize();
    return result;
}

// ============================================================
// Fused kernel wrapper
// ============================================================
template<int Arity>
std::vector<Tuple> run_fused_sort_dedup(SoARelation& in) {
    SoARelation out;
    out.alloc(in.count, Arity);
    uint32_t* d_count;
    cudaMalloc(&d_count, sizeof(uint32_t));
    cudaMemset(d_count, 0, sizeof(uint32_t));

    // Identity column order
    int h_col_order[Arity];
    for (int i = 0; i < Arity; i++) h_col_order[i] = i;
    static int* d_col_order = nullptr;
    if (!d_col_order) {
        cudaMalloc(&d_col_order, Arity * sizeof(int));
    }
    cudaMemcpy(d_col_order, h_col_order, Arity * sizeof(int), cudaMemcpyHostToDevice);

    tiny_sort_dedup_kernel<uint32_t, Arity><<<1, kSortBlock>>>(
        in.d_data, in.count, in.stride,
        d_col_order,
        out.d_data, d_count, out.stride);
    cudaDeviceSynchronize();

    uint32_t h_count;
    cudaMemcpy(&h_count, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    out.count = h_count;

    auto result = out.to_host();
    result.resize(h_count);
    out.free();
    cudaFree(d_count);
    return result;
}

// ============================================================
// Data generation
// ============================================================
std::vector<Tuple> gen_tuples(int count, int arity, float self_dup_ratio,
                               std::mt19937& rng, int value_range = 200000) {
    int num_unique = std::max((int)(count * (1.0f - self_dup_ratio)), 1);
    std::vector<Tuple> tuples(count, Tuple(arity));
    for (int i = 0; i < num_unique; i++)
        for (int j = 0; j < arity; j++)
            tuples[i][j] = rng() % value_range;
    for (int i = num_unique; i < count; i++) {
        int src = rng() % num_unique;
        tuples[i] = tuples[src];
    }
    return tuples;
}

SoARelation tuples_to_soa(const std::vector<Tuple>& tuples, int arity) {
    SoARelation rel;
    rel.alloc(tuples.size(), arity);
    std::vector<uint32_t> h(arity * tuples.size());
    for (size_t i = 0; i < tuples.size(); i++)
        for (int j = 0; j < arity; j++)
            h[j * tuples.size() + i] = tuples[i][j];
    cudaMemcpy(rel.d_data, h.data(), arity * tuples.size() * sizeof(uint32_t),
               cudaMemcpyHostToDevice);
    return rel;
}

// ============================================================
// Test harness
// ============================================================

bool verify_sorted(const std::vector<Tuple>& tuples) {
    for (size_t i = 1; i < tuples.size(); i++)
        if (tuples[i] < tuples[i-1]) return false;
    return true;
}

bool verify_deduped(const std::vector<Tuple>& tuples) {
    for (size_t i = 1; i < tuples.size(); i++)
        if (tuples[i] == tuples[i-1]) return false;
    return true;
}

bool verify_equal(const std::vector<Tuple>& a, const std::vector<Tuple>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++)
        if (a[i] != b[i]) return false;
    return true;
}

int total_tests = 0, passed_tests = 0;
void check(bool ok, const char* name) {
    total_tests++;
    if (ok) { passed_tests++; }
    else { printf("  FAIL: %s\n", name); }
}

int main() {
    std::mt19937 rng(42);

    // ==========================================================
    // Test 1: Basic correctness — count, sorted, deduped
    // ==========================================================
    printf("=== Test 1: Basic correctness ===\n");
    {
        struct TC { int n; float dup; int arity; const char* label; };
        TC cases[] = {
            {6,    0.0, 4, "6 unique, arity 4"},
            {6,    0.5, 4, "6 with 50%% dup, arity 4"},
            {20,   0.3, 4, "20 with 30%% dup, arity 4"},
            {20,   0.5, 3, "20 with 50%% dup, arity 3"},
            {100,  0.3, 4, "100 with 30%% dup, arity 4"},
            {500,  0.2, 4, "500 with 20%% dup, arity 4"},
            {1000, 0.2, 4, "1000 with 20%% dup, arity 4"},
            {1000, 0.5, 3, "1000 with 50%% dup, arity 3"},
        };
        for (auto& tc : cases) {
            auto input = gen_tuples(tc.n, tc.arity, tc.dup, rng);
            auto soa = tuples_to_soa(input, tc.arity);
            auto cpu_ref = cpu_sort_dedup(input);
            std::vector<Tuple> fused_result;
            if (tc.arity == 3) fused_result = run_fused_sort_dedup<3>(soa);
            else fused_result = run_fused_sort_dedup<4>(soa);

            printf("  %s: count=%zu (ref=%zu) ", tc.label, fused_result.size(), cpu_ref.size());
            check(fused_result.size() == cpu_ref.size(), "count match");
            check(verify_sorted(fused_result), "sorted");
            check(verify_deduped(fused_result), "no dups");
            check(verify_equal(fused_result, cpu_ref), "data match");
            printf("%s\n", (fused_result.size() == cpu_ref.size() &&
                           verify_sorted(fused_result) &&
                           verify_equal(fused_result, cpu_ref)) ? "OK" : "FAIL");
            soa.free();
        }
    }

    // ==========================================================
    // Test 2: Sort key edge cases — tuples differing only in col2/col3
    // ==========================================================
    printf("\n=== Test 2: Sort key edge cases ===\n");
    {
        // Tuples with same first 2 cols, different col2/col3
        std::vector<Tuple> input = {
            {1, 2, 100, 200},
            {1, 2, 50, 300},
            {1, 2, 100, 100},
            {1, 2, 50, 200},
            {1, 2, 100, 200},  // dup of [0]
            {2, 1, 10, 20},
        };
        auto soa = tuples_to_soa(input, 4);
        auto cpu_ref = cpu_sort_dedup(input);
        auto fused = run_fused_sort_dedup<4>(soa);
        printf("  Same first-2-cols: count=%zu (ref=%zu) ", fused.size(), cpu_ref.size());
        check(verify_equal(fused, cpu_ref), "col2/col3 sort correctness");
        printf("%s\n", verify_equal(fused, cpu_ref) ? "OK" : "FAIL");
        if (!verify_equal(fused, cpu_ref)) {
            printf("    Fused output:\n");
            for (auto& t : fused) printf("      (%u,%u,%u,%u)\n", t[0],t[1],t[2],t[3]);
            printf("    CPU ref:\n");
            for (auto& t : cpu_ref) printf("      (%u,%u,%u,%u)\n", t[0],t[1],t[2],t[3]);
        }
        soa.free();
    }

    // ==========================================================
    // Test 3: End-to-end — sort+dedup then set_difference against FULL
    // ==========================================================
    printf("\n=== Test 3: End-to-end with set_difference ===\n");
    {
        // Create a sorted FULL relation
        auto full_tuples = gen_tuples(5000, 4, 0.0, rng);
        full_tuples = cpu_sort_dedup(full_tuples);

        // Create NEW with some dups from FULL + self-dups + new tuples
        int new_count = 50;
        std::vector<Tuple> new_tuples;
        // 10 from FULL (should be filtered by set_diff)
        for (int i = 0; i < 10; i++)
            new_tuples.push_back(full_tuples[rng() % full_tuples.size()]);
        // 5 self-dups
        for (int i = 0; i < 5; i++)
            new_tuples.push_back(new_tuples[rng() % new_tuples.size()]);
        // 35 new unique
        for (int i = 0; i < 35; i++) {
            Tuple t(4);
            for (int j = 0; j < 4; j++) t[j] = 300000 + rng() % 100000;
            new_tuples.push_back(t);
        }

        // CPU reference: sort+dedup NEW, then set_diff against FULL
        auto cpu_sorted_new = cpu_sort_dedup(new_tuples);
        auto cpu_delta = cpu_set_difference(cpu_sorted_new, full_tuples);

        // Fused: sort+dedup NEW, then use thrust set_diff against FULL
        auto soa_new = tuples_to_soa(new_tuples, 4);
        auto fused_sorted = run_fused_sort_dedup<4>(soa_new);

        // Verify fused sorted matches cpu sorted
        check(verify_equal(fused_sorted, cpu_sorted_new), "sorted NEW matches CPU");

        // Now do set_diff on fused output against FULL
        auto fused_delta = cpu_set_difference(fused_sorted, full_tuples);
        check(verify_equal(fused_delta, cpu_delta), "delta after set_diff matches");
        printf("  NEW=%d, sorted+deduped=%zu, delta=%zu (ref=%zu) %s\n",
               new_count, fused_sorted.size(), fused_delta.size(), cpu_delta.size(),
               verify_equal(fused_delta, cpu_delta) ? "OK" : "FAIL");
        soa_new.free();
    }

    // ==========================================================
    // Test 4: Multi-iteration simulation
    // ==========================================================
    printf("\n=== Test 4: Multi-iteration fixpoint simulation ===\n");
    {
        std::set<Tuple> full_set;
        int num_iters = 20;
        bool all_ok = true;

        for (int iter = 0; iter < num_iters; iter++) {
            // Generate small NEW batch
            int batch_size = 5 + rng() % 30;
            auto new_tuples = gen_tuples(batch_size, 4, 0.3, rng, 500);

            // Fused sort+dedup
            auto soa_new = tuples_to_soa(new_tuples, 4);
            auto fused_sorted = run_fused_sort_dedup<4>(soa_new);

            // CPU sort+dedup
            auto cpu_sorted = cpu_sort_dedup(new_tuples);

            if (!verify_equal(fused_sorted, cpu_sorted)) {
                printf("  Iter %d: sort+dedup MISMATCH!\n", iter);
                all_ok = false;
            }

            // Set difference against accumulated FULL
            std::vector<Tuple> full_vec(full_set.begin(), full_set.end());
            auto fused_delta = cpu_set_difference(fused_sorted, full_vec);
            auto cpu_delta = cpu_set_difference(cpu_sorted, full_vec);

            if (!verify_equal(fused_delta, cpu_delta)) {
                printf("  Iter %d: delta MISMATCH! fused=%zu cpu=%zu\n",
                       iter, fused_delta.size(), cpu_delta.size());
                all_ok = false;
            }

            // Merge delta into FULL
            for (auto& t : fused_delta)
                full_set.insert(t);

            soa_new.free();
        }
        check(all_ok, "all iterations match");
        printf("  %d iterations, final FULL size=%zu %s\n",
               num_iters, full_set.size(), all_ok ? "OK" : "FAIL");
    }

    // ==========================================================
    // Test 5: Performance comparison
    // ==========================================================
    printf("\n=== Test 5: Performance ===\n");
    {
        struct PC { int n; float dup; int arity; const char* label; };
        PC cases[] = {
            {6,    0.3, 4, "6 tuples, arity 4"},
            {20,   0.3, 4, "20 tuples, arity 4"},
            {100,  0.3, 4, "100 tuples, arity 4"},
            {500,  0.2, 4, "500 tuples, arity 4"},
            {1000, 0.2, 4, "1000 tuples, arity 4"},
            {50,   0.2, 3, "50 tuples, arity 3"},
        };
        int warmup = 500, iters = 5000;

        for (auto& pc : cases) {
            auto input = gen_tuples(pc.n, pc.arity, pc.dup, rng);
            auto soa = tuples_to_soa(input, pc.arity);

            // Warmup
            for (int i = 0; i < warmup; i++) {
                if (pc.arity == 3) run_fused_sort_dedup<3>(soa);
                else run_fused_sort_dedup<4>(soa);
            }

            // Benchmark fused
            cudaDeviceSynchronize();
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; i++) {
                if (pc.arity == 3) run_fused_sort_dedup<3>(soa);
                else run_fused_sort_dedup<4>(soa);
            }
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            double fused_us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
                              / 1e3 / iters;

            // Benchmark thrust
            SoARelation out; out.alloc(pc.n, pc.arity);
            for (int i = 0; i < warmup; i++) {
                if (pc.arity == 3) run_thrust_sort_dedup<3>(soa);
                else run_thrust_sort_dedup<4>(soa);
            }
            cudaDeviceSynchronize();
            t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; i++) {
                if (pc.arity == 3) run_thrust_sort_dedup<3>(soa);
                else run_thrust_sort_dedup<4>(soa);
            }
            cudaDeviceSynchronize();
            t1 = std::chrono::high_resolution_clock::now();
            double thrust_us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
                               / 1e3 / iters;

            printf("  %s: fused=%.1fus thrust=%.1fus speedup=%.1fx\n",
                   pc.label, fused_us, thrust_us, thrust_us / fused_us);
            soa.free(); out.free();
        }
    }

    printf("\n=== Results: %d/%d tests passed ===\n", passed_tests, total_tests);
    return (passed_tests == total_tests) ? 0 : 1;
}
