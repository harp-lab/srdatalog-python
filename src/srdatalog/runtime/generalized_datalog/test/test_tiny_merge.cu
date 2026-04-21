/**
 * Benchmark: fused probe + sort_dedup + parallel merge
 *
 * Full pipeline replacing rebuild_index + compute_delta + merge_index:
 *   Kernel 1 (probe): binary search NEW against FULL → unsorted survivors
 *   Kernel 2 (sort_dedup): CUB sort + dedup → sorted DELTA
 *   Kernel 3 (tiny_merge): merge tiny sorted DELTA into large sorted FULL
 *     - Each DELTA tuple binary-searches its rank in FULL
 *     - Parallel scatter: FULL elements + DELTA elements → new merged array
 *     - 1 kernel vs thrust::merge (3-4 CUB launches)
 *
 * Total: 3 kernels vs 12-35 launches in current pipeline.
 */

#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <set>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/merge.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/set_operations.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "gpu/runtime/jit/tiny_sort_dedup.h"
using namespace SRDatalog::GPU::JIT;

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
    std::vector<std::vector<uint32_t>> to_host() const {
        std::vector<uint32_t> h(arity * stride);
        cudaMemcpy(h.data(), d_data, arity * stride * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        std::vector<std::vector<uint32_t>> tuples(count, std::vector<uint32_t>(arity));
        for (uint32_t i = 0; i < count; i++)
            for (int j = 0; j < arity; j++)
                tuples[i][j] = h[j * stride + i];
        return tuples;
    }
    void from_tuples(const std::vector<std::vector<uint32_t>>& tuples) {
        int n = tuples.size();
        int ar = tuples.empty() ? arity : tuples[0].size();
        if (!d_data || (uint32_t)n > stride) { free(); alloc(n, ar); }
        count = n; arity = ar;
        std::vector<uint32_t> h(arity * stride, 0);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < arity; j++)
                h[j * stride + i] = tuples[i][j];
        cudaMemcpy(d_data, h.data(), arity * stride * sizeof(uint32_t), cudaMemcpyHostToDevice);
    }
};

using Tuple = std::vector<uint32_t>;

// ============================================================
// Kernel 1: probe NEW against sorted FULL
// ============================================================
template<int Arity>
__device__ __forceinline__ int compare_soa(
    const uint32_t* a, uint32_t ai, uint32_t as,
    const uint32_t* b, uint32_t bi, uint32_t bs) {
    #pragma unroll
    for (int c = 0; c < Arity; c++) {
        uint32_t va = a[c * as + ai], vb = b[c * bs + bi];
        if (va < vb) return -1;
        if (va > vb) return 1;
    }
    return 0;
}

template<int Arity>
__global__ void probe_kernel(
    const uint32_t* new_data, uint32_t new_count, uint32_t new_stride,
    const uint32_t* full_data, uint32_t full_count, uint32_t full_stride,
    uint32_t* surv_data, uint32_t* surv_count, uint32_t surv_stride) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= new_count) return;
    int lo = 0, hi = (int)full_count - 1;
    bool found = false;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        int cmp = compare_soa<Arity>(new_data, tid, new_stride, full_data, mid, full_stride);
        if (cmp == 0) { found = true; break; }
        if (cmp < 0) hi = mid - 1; else lo = mid + 1;
    }
    if (!found) {
        uint32_t pos = atomicAdd(surv_count, 1u);
        #pragma unroll
        for (int c = 0; c < Arity; c++)
            surv_data[c * surv_stride + pos] = new_data[c * new_stride + tid];
    }
}

// ============================================================
// Kernel 3: tiny merge — merge small sorted DELTA into large sorted FULL
//
// Each thread handles a range of output positions (grid-stride).
// For output position i:
//   - Binary search delta_count to find how many DELTA elements go before i
//   - The element at position i is either from FULL or DELTA
//
// Uses the merge-path approach: for output position i, find the split
// (a, b) where a + b = i, a elements from FULL and b from DELTA,
// such that FULL[a-1] <= DELTA[b] and DELTA[b-1] <= FULL[a].
//
// Since DELTA is tiny, we binary-search in DELTA for each output position.
// ============================================================
template<int Arity>
__global__ void tiny_merge_kernel(
    const uint32_t* __restrict__ full_data, uint32_t full_count, uint32_t full_stride,
    const uint32_t* __restrict__ delta_data, uint32_t delta_count, uint32_t delta_stride,
    uint32_t* __restrict__ out_data, uint32_t out_stride) {

    uint32_t total = full_count + delta_count;

    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += blockDim.x * gridDim.x) {
        // Merge path: find split (a, b) where a + b = i
        // a = elements from FULL before position i
        // b = elements from DELTA before position i
        // Binary search for the correct split in DELTA
        //
        // We want the largest b such that:
        //   b <= delta_count AND
        //   b <= i AND
        //   (b == 0 OR DELTA[b-1] <= FULL[i-b])
        //
        // Equivalently: binary search for b in [0, min(delta_count, i+1)]
        // where DELTA[b] > FULL[i-b] (upper bound)

        uint32_t lo = (i >= full_count) ? (i - full_count) : 0;
        uint32_t hi = (i < delta_count) ? i : delta_count;

        while (lo < hi) {
            uint32_t mid = (lo + hi) / 2;
            uint32_t a = i - mid;  // corresponding FULL index

            // Compare DELTA[mid] vs FULL[a]
            // If DELTA[mid] < FULL[a], we need more from DELTA (increase b)
            // If DELTA[mid] >= FULL[a], we need less from DELTA (decrease b)
            int cmp;
            if (a >= full_count) {
                cmp = 1;  // FULL exhausted, take from DELTA
            } else {
                cmp = compare_soa<Arity>(delta_data, mid, delta_stride,
                                          full_data, a, full_stride);
            }

            if (cmp < 0) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        uint32_t b = lo;      // elements from DELTA
        uint32_t a = i - b;   // elements from FULL

        // Position i gets element from DELTA[b] if b < delta_count AND
        // (a >= full_count OR DELTA[b] < FULL[a])
        // Note: strict < gives FULL priority on equal (stable: FULL before DELTA)
        bool from_delta;
        if (b >= delta_count) {
            from_delta = false;
        } else if (a >= full_count) {
            from_delta = true;
        } else {
            int cmp = compare_soa<Arity>(delta_data, b, delta_stride,
                                          full_data, a, full_stride);
            from_delta = (cmp < 0);
        }

        if (from_delta) {
            #pragma unroll
            for (int c = 0; c < Arity; c++)
                out_data[c * out_stride + i] = delta_data[c * delta_stride + b];
        } else {
            #pragma unroll
            for (int c = 0; c < Arity; c++)
                out_data[c * out_stride + i] = full_data[c * full_stride + a];
        }
    }
}

// ============================================================
// Full fused pipeline
// ============================================================
template<int Arity>
uint32_t run_fused_pipeline(SoARelation& new_rel, SoARelation& full,
                            SoARelation& merged_out) {
    // Step 1: probe
    SoARelation survivors;
    survivors.alloc(new_rel.count, Arity);
    uint32_t* d_surv_count;
    cudaMalloc(&d_surv_count, sizeof(uint32_t));
    cudaMemset(d_surv_count, 0, sizeof(uint32_t));

    int block = 256;
    int grid = std::max((int)((new_rel.count + block - 1) / block), 1);
    probe_kernel<Arity><<<grid, block>>>(
        new_rel.d_data, new_rel.count, new_rel.stride,
        full.d_data, full.count, full.stride,
        survivors.d_data, d_surv_count, survivors.stride);

    uint32_t h_surv;
    cudaDeviceSynchronize();
    cudaMemcpy(&h_surv, d_surv_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    if (h_surv == 0) {
        survivors.free(); cudaFree(d_surv_count);
        merged_out = full;
        return 0;
    }

    // Step 2: sort + self-dedup → sorted DELTA
    SoARelation delta;
    delta.alloc(h_surv, Arity);
    uint32_t* d_delta_count;
    cudaMalloc(&d_delta_count, sizeof(uint32_t));
    cudaMemset(d_delta_count, 0, sizeof(uint32_t));

    // Identity column order
    static int* d_col_order = nullptr;
    if (!d_col_order) {
        int h_col[] = {0, 1, 2, 3};
        cudaMalloc(&d_col_order, 4 * sizeof(int));
        cudaMemcpy(d_col_order, h_col, 4 * sizeof(int), cudaMemcpyHostToDevice);
    }

    tiny_sort_dedup_kernel<uint32_t, Arity><<<1, kTinySortBlock>>>(
        survivors.d_data, h_surv, survivors.stride,
        d_col_order,
        delta.d_data, d_delta_count, delta.stride);

    uint32_t h_delta;
    cudaDeviceSynchronize();
    cudaMemcpy(&h_delta, d_delta_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    delta.count = h_delta;

    if (h_delta == 0) {
        survivors.free(); delta.free();
        cudaFree(d_surv_count); cudaFree(d_delta_count);
        merged_out = full;
        return 0;
    }

    // Step 3: tiny merge — single kernel
    uint32_t total = full.count + h_delta;
    merged_out.alloc(total, Arity);

    int merge_block = 256;
    int merge_grid = std::min((int)((total + merge_block - 1) / merge_block), 128);
    tiny_merge_kernel<Arity><<<merge_grid, merge_block>>>(
        full.d_data, full.count, full.stride,
        delta.d_data, h_delta, delta.stride,
        merged_out.d_data, merged_out.stride);
    merged_out.count = total;

    cudaDeviceSynchronize();
    survivors.free(); delta.free();
    cudaFree(d_surv_count); cudaFree(d_delta_count);
    return h_delta;
}

// ============================================================
// Thrust baseline: full standard pipeline
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
uint32_t run_thrust_pipeline(SoARelation& new_rel, SoARelation& full,
                             SoARelation& merged_out) {
    uint32_t N = new_rel.count;
    // Sort
    thrust::device_vector<uint32_t> perm(N);
    thrust::sequence(perm.begin(), perm.end());
    SoAComparator<Arity> cmp{new_rel.d_data, new_rel.stride};
    thrust::sort(perm.begin(), perm.end(), cmp);
    // Gather sorted
    thrust::device_vector<uint32_t> sorted(Arity * N);
    for (int c = 0; c < Arity; c++)
        thrust::gather(perm.begin(), perm.end(),
                       thrust::device_pointer_cast(new_rel.d_data + c * new_rel.stride),
                       sorted.begin() + c * N);
    // Unique
    thrust::device_vector<uint32_t> uperm(N);
    thrust::sequence(uperm.begin(), uperm.end());
    SoAEqual<Arity> eq{thrust::raw_pointer_cast(sorted.data()), N};
    auto uend = thrust::unique(uperm.begin(), uperm.end(), eq);
    uint32_t uniq = uend - uperm.begin();

    // Gather unique into contiguous buffer
    SoARelation sorted_new; sorted_new.alloc(uniq, Arity);
    for (int c = 0; c < Arity; c++) {
        thrust::device_vector<uint32_t> tmp(uniq);
        thrust::gather(uperm.begin(), uperm.begin() + uniq,
                       sorted.begin() + c * N, tmp.begin());
        cudaMemcpy(sorted_new.d_data + c * sorted_new.stride,
                   thrust::raw_pointer_cast(tmp.data()),
                   uniq * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    }
    sorted_new.count = uniq;

    // Set difference + merge via zip iterator
    auto mk4 = [](uint32_t* data, uint32_t stride) {
        return thrust::make_zip_iterator(thrust::make_tuple(
            thrust::device_pointer_cast(data + 0 * stride),
            thrust::device_pointer_cast(data + 1 * stride),
            thrust::device_pointer_cast(data + 2 * stride),
            thrust::device_pointer_cast(data + 3 * stride)));
    };

    SoARelation delta_buf; delta_buf.alloc(uniq, Arity);
    auto dend = thrust::set_difference(thrust::device,
        mk4(sorted_new.d_data, sorted_new.stride),
        mk4(sorted_new.d_data, sorted_new.stride) + uniq,
        mk4(full.d_data, full.stride),
        mk4(full.d_data, full.stride) + full.count,
        mk4(delta_buf.d_data, delta_buf.stride));
    uint32_t dc = dend - mk4(delta_buf.d_data, delta_buf.stride);
    delta_buf.count = dc;

    uint32_t total = full.count + dc;
    merged_out.alloc(total, Arity);
    thrust::merge(thrust::device,
        mk4(full.d_data, full.stride), mk4(full.d_data, full.stride) + full.count,
        mk4(delta_buf.d_data, delta_buf.stride), mk4(delta_buf.d_data, delta_buf.stride) + dc,
        mk4(merged_out.d_data, merged_out.stride));
    merged_out.count = total;

    sorted_new.free(); delta_buf.free();
    cudaDeviceSynchronize();
    return dc;
}

// ============================================================
// CPU reference
// ============================================================
std::vector<Tuple> cpu_sort_dedup(const std::vector<Tuple>& in) {
    auto t = in; std::sort(t.begin(), t.end());
    t.erase(std::unique(t.begin(), t.end()), t.end()); return t;
}

// ============================================================
// Test harness
// ============================================================
int total_tests = 0, passed = 0;
void check(bool ok, const char* name) {
    total_tests++; if (ok) passed++; else printf("  FAIL: %s\n", name);
}

std::vector<Tuple> gen_tuples(int n, int ar, float dup, std::mt19937& rng, int range = 200000) {
    int nu = std::max((int)(n * (1.0f - dup)), 1);
    std::vector<Tuple> t(n, Tuple(ar));
    for (int i = 0; i < nu; i++) for (int j = 0; j < ar; j++) t[i][j] = rng() % range;
    for (int i = nu; i < n; i++) t[i] = t[rng() % nu];
    return t;
}

int main() {
    std::mt19937 rng(42);

    // ==========================================================
    // Test 1: End-to-end correctness
    // ==========================================================
    printf("=== Test 1: End-to-end correctness ===\n");
    {
        struct TC { int full_sz, new_sz; float dup_full, self_dup; const char* label; };
        TC cases[] = {
            {5000,  6,   0.3, 0.2, "6 new, 5K full"},
            {5000,  20,  0.3, 0.3, "20 new, 5K full"},
            {5000,  100, 0.3, 0.2, "100 new, 5K full"},
            {5000,  500, 0.2, 0.1, "500 new, 5K full"},
            {50000, 20,  0.3, 0.2, "20 new, 50K full"},
            {50000, 100, 0.2, 0.1, "100 new, 50K full"},
        };
        for (auto& tc : cases) {
            auto full_t = gen_tuples(tc.full_sz, 4, 0.0, rng);
            full_t = cpu_sort_dedup(full_t);
            SoARelation full; full.from_tuples(full_t);

            std::vector<Tuple> new_t;
            int nf = (int)(tc.new_sz * tc.dup_full);
            int ns = (int)(tc.new_sz * tc.self_dup);
            for (int i = 0; i < tc.new_sz - nf - ns; i++) {
                Tuple t(4); for (int j = 0; j < 4; j++) t[j] = 300000 + rng() % 100000;
                new_t.push_back(t);
            }
            for (int i = 0; i < nf; i++) new_t.push_back(full_t[rng() % full_t.size()]);
            for (int i = 0; i < ns && !new_t.empty(); i++) new_t.push_back(new_t[rng() % new_t.size()]);
            SoARelation new_rel; new_rel.from_tuples(new_t);

            // CPU reference
            auto cpu_sorted = cpu_sort_dedup(new_t);
            std::vector<Tuple> cpu_delta;
            std::set_difference(cpu_sorted.begin(), cpu_sorted.end(),
                                full_t.begin(), full_t.end(), std::back_inserter(cpu_delta));
            std::vector<Tuple> cpu_merged;
            std::merge(full_t.begin(), full_t.end(), cpu_delta.begin(), cpu_delta.end(),
                       std::back_inserter(cpu_merged));

            // Fused pipeline
            SoARelation merged;
            uint32_t dc = run_fused_pipeline<4>(new_rel, full, merged);
            auto fused_merged = (merged.d_data == full.d_data) ? full_t : merged.to_host();

            printf("  %s: delta=%u (ref=%zu) merged=%zu (ref=%zu) ",
                   tc.label, dc, cpu_delta.size(), fused_merged.size(), cpu_merged.size());
            check(dc == cpu_delta.size(), "delta count");
            check(fused_merged.size() == cpu_merged.size(), "merged count");
            check(fused_merged == cpu_merged, "merged data");
            printf("%s\n", (fused_merged == cpu_merged) ? "OK" : "FAIL");

            full.free(); new_rel.free();
            if (merged.d_data && merged.d_data != full.d_data) merged.free();
        }
    }

    // ==========================================================
    // Test 2: Multi-iteration fixpoint
    // ==========================================================
    printf("\n=== Test 2: Multi-iteration fixpoint ===\n");
    {
        auto init = gen_tuples(1000, 4, 0.0, rng, 500);
        init = cpu_sort_dedup(init);
        std::vector<Tuple> cpu_full = init;
        SoARelation gpu_full; gpu_full.from_tuples(init);

        int num_iters = 30;
        bool all_ok = true;
        for (int iter = 0; iter < num_iters; iter++) {
            auto new_t = gen_tuples(5 + rng() % 30, 4, 0.3, rng, 500);
            SoARelation gpu_new; gpu_new.from_tuples(new_t);

            SoARelation merged;
            run_fused_pipeline<4>(gpu_new, gpu_full, merged);

            auto cpu_sorted = cpu_sort_dedup(new_t);
            std::vector<Tuple> cpu_delta;
            std::set_difference(cpu_sorted.begin(), cpu_sorted.end(),
                                cpu_full.begin(), cpu_full.end(), std::back_inserter(cpu_delta));
            std::vector<Tuple> cpu_merged;
            std::merge(cpu_full.begin(), cpu_full.end(),
                       cpu_delta.begin(), cpu_delta.end(), std::back_inserter(cpu_merged));

            // Always read from GPU to verify
            SoARelation& actual_full = (merged.d_data && merged.d_data != gpu_full.d_data) ? merged : gpu_full;
            actual_full.count = (merged.d_data && merged.d_data != gpu_full.d_data) ? merged.count : gpu_full.count;
            auto fused_merged = actual_full.to_host();
            if (fused_merged != cpu_merged) {
                printf("  Iter %d: MISMATCH fused=%zu cpu=%zu\n",
                       iter, fused_merged.size(), cpu_merged.size());
                all_ok = false;
            }

            cpu_full = cpu_merged;
            if (merged.d_data && merged.d_data != gpu_full.d_data) gpu_full.free();
            gpu_full = merged;
            gpu_new.free();
        }
        check(all_ok, "all iterations match");
        printf("  %d iterations, final FULL=%zu %s\n",
               num_iters, cpu_full.size(), all_ok ? "OK" : "FAIL");
        gpu_full.free();
    }

    // ==========================================================
    // Test 3: Performance
    // ==========================================================
    printf("\n=== Test 3: Performance (full pipeline) ===\n");
    {
        struct PC { int full_sz, new_sz; const char* label; };
        PC cases[] = {
            {65000, 6,    "6 new, 65K full"},
            {65000, 20,   "20 new, 65K full"},
            {65000, 50,   "50 new, 65K full"},
            {65000, 100,  "100 new, 65K full"},
            {65000, 500,  "500 new, 65K full"},
        };
        int warmup = 200, iters = 1000;

        for (auto& pc : cases) {
            auto full_t = gen_tuples(pc.full_sz, 4, 0.0, rng);
            full_t = cpu_sort_dedup(full_t);
            SoARelation full; full.from_tuples(full_t);
            auto new_t = gen_tuples(pc.new_sz, 4, 0.3, rng);
            SoARelation new_rel; new_rel.from_tuples(new_t);

            // Warmup + bench fused
            for (int i = 0; i < warmup; i++) {
                SoARelation m; run_fused_pipeline<4>(new_rel, full, m);
                if (m.d_data && m.d_data != full.d_data) m.free();
            }
            cudaDeviceSynchronize();
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; i++) {
                SoARelation m; run_fused_pipeline<4>(new_rel, full, m);
                if (m.d_data && m.d_data != full.d_data) m.free();
            }
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            double fused_us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()
                              / 1e3 / iters;

            // Warmup + bench thrust
            for (int i = 0; i < warmup; i++) {
                SoARelation m; run_thrust_pipeline<4>(new_rel, full, m); m.free();
            }
            cudaDeviceSynchronize();
            t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; i++) {
                SoARelation m; run_thrust_pipeline<4>(new_rel, full, m); m.free();
            }
            cudaDeviceSynchronize();
            t1 = std::chrono::high_resolution_clock::now();
            double thrust_us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()
                               / 1e3 / iters;

            printf("  %s: fused=%.0fus thrust=%.0fus speedup=%.1fx\n",
                   pc.label, fused_us, thrust_us, thrust_us / fused_us);
            full.free(); new_rel.free();
        }
    }

    printf("\n=== Results: %d/%d tests passed ===\n", passed, total_tests);
    return (passed == total_tests) ? 0 : 1;
}
