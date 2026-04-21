/**
 * Benchmark: multi-order fused sort+dedup in one kernel
 *
 * rebuild_index is called for EACH index ordering. For a relation with
 * 3 index orderings (e.g., [0,1,2,3], [2,1,0,3], [1,2,0,3]), the current
 * code launches 3 × (sort + gather + unique) = 30+ kernel launches.
 *
 * This kernel does ALL orderings in one launch:
 *   1. Load tuples into shared memory
 *   2. For each ordering: pack sort key → CUB BlockRadixSort → gather → dedup
 *   3. Write all sorted+deduped outputs
 *
 * Key constraint: shared memory limits how many orderings × tuples we can handle.
 * RTX 6000 Ada: 100KB shared memory per block.
 * Per ordering: kMaxItems × Arity × 4 bytes = 1024 × 4 × 4 = 16KB
 * So ~6 orderings fit in 100KB.
 */

#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

static constexpr int kSortBlock = 256;
static constexpr int kItemsPerThread = 4;
static constexpr int kMaxItems = kSortBlock * kItemsPerThread;  // 1024
static constexpr int kMaxOrders = 4;  // max orderings per kernel
static constexpr int kMaxArity = 4;

// ============================================================
// Ordering descriptor (passed to kernel)
// ============================================================
struct IndexOrder {
    int cols[kMaxArity];  // column permutation, e.g., {2,1,0,3}
    int arity;
};

// ============================================================
// Device helpers
// ============================================================
template<int Arity>
__device__ __forceinline__ int compare_reordered(
    const uint32_t* data, uint32_t ai, uint32_t as,
    uint32_t bi,
    const int* col_order
) {
    #pragma unroll
    for (int i = 0; i < Arity; i++) {
        int c = col_order[i];
        uint32_t va = data[c * as + ai], vb = data[c * as + bi];
        if (va < vb) return -1;
        if (va > vb) return 1;
    }
    return 0;
}

// ============================================================
// Multi-order sort+dedup kernel
//
// Input:  unsorted SoA tuples (shared across all orderings)
// Output: for each ordering, sorted+deduped SoA + count
//
// Strategy:
//   - Load input once into shared memory
//   - For each ordering sequentially:
//     a. Pack sort key from column permutation
//     b. CUB BlockRadixSort
//     c. Gather + adjacent-unique
//     d. Compact to output via atomicAdd
// ============================================================
template<int Arity, int NumOrders>
__global__ void multi_sort_dedup_kernel(
    const uint32_t* __restrict__ in_data, uint32_t in_count, uint32_t in_stride,
    // Per-ordering outputs (array of pointers)
    uint32_t* __restrict__ out_data_ptrs[kMaxOrders],
    uint32_t* __restrict__ out_count_ptrs[kMaxOrders],
    uint32_t out_strides[kMaxOrders],
    // Ordering descriptors
    IndexOrder orders[kMaxOrders]
) {
    // Shared memory layout:
    // - s_input: original tuples (Arity × kMaxItems) — persistent across orderings
    // - s_sorted: workspace for current ordering (Arity × kMaxItems) — reused
    using BlockSortT = cub::BlockRadixSort<uint64_t, kSortBlock, kItemsPerThread, uint32_t>;
    __shared__ union {
        typename BlockSortT::TempStorage sort;
    } temp_storage;

    __shared__ uint32_t s_input[Arity * kMaxItems];
    __shared__ uint32_t s_sorted[Arity * kMaxItems];

    uint32_t tid = threadIdx.x;

    // Step 1: Load input into shared memory (once, reused across orderings)
    #pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
        uint32_t idx = tid * kItemsPerThread + i;
        if (idx < in_count) {
            #pragma unroll
            for (int c = 0; c < Arity; c++)
                s_input[c * kMaxItems + idx] = in_data[c * in_stride + idx];
        } else {
            #pragma unroll
            for (int c = 0; c < Arity; c++)
                s_input[c * kMaxItems + idx] = 0xFFFFFFFFu;
        }
    }
    __syncthreads();

    // Step 2: For each ordering, sort + dedup + write output
    for (int ord = 0; ord < NumOrders; ord++) {
        const int* col_order = orders[ord].cols;

        // 2a. Pack sort key using this ordering's column permutation
        uint64_t my_keys[kItemsPerThread];
        uint32_t my_vals[kItemsPerThread];

        #pragma unroll
        for (int i = 0; i < kItemsPerThread; i++) {
            uint32_t idx = tid * kItemsPerThread + i;
            if (idx < in_count) {
                uint32_t c0 = s_input[col_order[0] * kMaxItems + idx];
                uint32_t c1 = (Arity >= 2) ? s_input[col_order[1] * kMaxItems + idx] : 0u;
                my_keys[i] = ((uint64_t)c0 << 32) | c1;
                my_vals[i] = idx;  // index into s_input
            } else {
                my_keys[i] = 0xFFFFFFFFFFFFFFFFull;
                my_vals[i] = 0xFFFFFFFFu;
            }
        }

        // 2b. CUB block sort
        BlockSortT(temp_storage.sort).Sort(my_keys, my_vals);
        __syncthreads();

        // 2c. Gather sorted tuples in this ordering's column order
        #pragma unroll
        for (int i = 0; i < kItemsPerThread; i++) {
            uint32_t slot = tid * kItemsPerThread + i;
            uint32_t orig = my_vals[i];
            if (orig < in_count) {
                #pragma unroll
                for (int c = 0; c < Arity; c++)
                    s_sorted[c * kMaxItems + slot] = s_input[col_order[c] * kMaxItems + orig];
            } else {
                #pragma unroll
                for (int c = 0; c < Arity; c++)
                    s_sorted[c * kMaxItems + slot] = 0xFFFFFFFFu;
            }
        }
        __syncthreads();

        // 2d. Adjacent-unique + compact to output
        uint32_t* out_data = out_data_ptrs[ord];
        uint32_t* out_count = out_count_ptrs[ord];
        uint32_t out_stride = out_strides[ord];

        #pragma unroll
        for (int i = 0; i < kItemsPerThread; i++) {
            uint32_t slot = tid * kItemsPerThread + i;
            if (slot >= in_count) continue;

            bool is_unique;
            if (slot == 0) {
                is_unique = true;
            } else {
                // Full arity comparison in sorted order
                is_unique = false;
                #pragma unroll
                for (int c = 0; c < Arity; c++) {
                    if (s_sorted[c * kMaxItems + slot] != s_sorted[c * kMaxItems + slot - 1]) {
                        is_unique = true;
                        break;
                    }
                }
            }

            if (is_unique) {
                uint32_t pos = atomicAdd(out_count, 1u);
                #pragma unroll
                for (int c = 0; c < Arity; c++)
                    out_data[c * out_stride + pos] = s_sorted[c * kMaxItems + slot];
            }
        }
        __syncthreads();  // sync before reusing s_sorted for next ordering
    }
}

// ============================================================
// Thrust baseline: separate sort+dedup per ordering
// ============================================================
template<int Arity>
struct ReorderedCmp {
    const uint32_t* data; uint32_t stride;
    int cols[kMaxArity];
    __host__ __device__ bool operator()(uint32_t a, uint32_t b) const {
        for (int i = 0; i < Arity; i++) {
            uint32_t va = data[cols[i] * stride + a], vb = data[cols[i] * stride + b];
            if (va < vb) return true;
            if (va > vb) return false;
        }
        return false;
    }
};

template<int Arity>
struct ReorderedEq {
    const uint32_t* data; uint32_t stride;
    int cols[kMaxArity];
    __host__ __device__ bool operator()(uint32_t a, uint32_t b) const {
        for (int i = 0; i < Arity; i++)
            if (data[cols[i] * stride + a] != data[cols[i] * stride + b]) return false;
        return true;
    }
};

template<int Arity>
void run_thrust_multi(uint32_t* d_data, uint32_t count, uint32_t stride,
                      IndexOrder* orders, int num_orders,
                      uint32_t** out_data, uint32_t* out_counts) {
    for (int ord = 0; ord < num_orders; ord++) {
        uint32_t N = count;
        thrust::device_vector<uint32_t> perm(N);
        thrust::sequence(perm.begin(), perm.end());

        ReorderedCmp<Arity> cmp{d_data, stride, {}};
        for (int i = 0; i < Arity; i++) cmp.cols[i] = orders[ord].cols[i];
        thrust::sort(perm.begin(), perm.end(), cmp);

        // Gather reordered columns
        thrust::device_vector<uint32_t> sorted(Arity * N);
        for (int c = 0; c < Arity; c++) {
            thrust::gather(perm.begin(), perm.end(),
                           thrust::device_pointer_cast(d_data + orders[ord].cols[c] * stride),
                           sorted.begin() + c * N);
        }

        // Unique
        thrust::device_vector<uint32_t> uniq_perm(N);
        thrust::sequence(uniq_perm.begin(), uniq_perm.end());
        ReorderedEq<Arity> eq{thrust::raw_pointer_cast(sorted.data()), N, {}};
        for (int i = 0; i < Arity; i++) eq.cols[i] = i;  // already reordered
        auto end = thrust::unique(uniq_perm.begin(), uniq_perm.end(), eq);
        out_counts[ord] = end - uniq_perm.begin();
    }
    cudaDeviceSynchronize();
}

// ============================================================
// CPU reference
// ============================================================
void cpu_multi_sort_dedup(uint32_t* d_data, uint32_t count, uint32_t stride, int arity,
                          IndexOrder* orders, int num_orders, uint32_t* ref_counts) {
    std::vector<uint32_t> h(arity * count);
    cudaMemcpy(h.data(), d_data, arity * count * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    for (int ord = 0; ord < num_orders; ord++) {
        std::vector<std::vector<uint32_t>> tuples(count, std::vector<uint32_t>(arity));
        for (uint32_t i = 0; i < count; i++)
            for (int j = 0; j < arity; j++)
                tuples[i][j] = h[orders[ord].cols[j] * count + i];
        std::sort(tuples.begin(), tuples.end());
        tuples.erase(std::unique(tuples.begin(), tuples.end()), tuples.end());
        ref_counts[ord] = tuples.size();
    }
}

// ============================================================
void create_data_with_dups(uint32_t* d_data, int count, int arity, float dup_ratio,
                           std::mt19937& rng) {
    int num_unique = std::max((int)(count * (1.0f - dup_ratio)), 1);
    std::vector<uint32_t> h(arity * count);
    for (int i = 0; i < num_unique; i++)
        for (int j = 0; j < arity; j++)
            h[j * count + i] = rng() % 200000;
    for (int i = num_unique; i < count; i++) {
        int src = rng() % num_unique;
        for (int j = 0; j < arity; j++)
            h[j * count + i] = h[j * count + src];
    }
    cudaMemcpy(d_data, h.data(), arity * count * sizeof(uint32_t), cudaMemcpyHostToDevice);
}

int main() {
    std::mt19937 rng(42);

    struct Config {
        int count;
        float dup_ratio;
        int arity;
        int num_orders;
        IndexOrder orders[kMaxOrders];
        const char* label;
    };

    Config configs[] = {
        {6, 0.3, 4, 3,
         {{{0,1,2,3}, 4}, {{2,1,0,3}, 4}, {{1,2,0,3}, 4}},
         "ddisasm tail: 6 tuples, 3 orders, arity 4"},
        {20, 0.3, 4, 3,
         {{{0,1,2,3}, 4}, {{2,1,0,3}, 4}, {{1,2,0,3}, 4}},
         "typical tail: 20 tuples, 3 orders, arity 4"},
        {50, 0.2, 3, 2,
         {{{0,1,2}, 3}, {{1,2,0}, 3}},
         "polonius tail: 50 tuples, 2 orders, arity 3"},
        {100, 0.3, 4, 4,
         {{{0,1,2,3}, 4}, {{2,1,0,3}, 4}, {{1,2,0,3}, 4}, {{3,0,1,2}, 4}},
         "medium: 100 tuples, 4 orders, arity 4"},
        {500, 0.2, 4, 3,
         {{{0,1,2,3}, 4}, {{2,1,0,3}, 4}, {{1,2,0,3}, 4}},
         "large: 500 tuples, 3 orders, arity 4"},
        {1000, 0.2, 4, 3,
         {{{0,1,2,3}, 4}, {{2,1,0,3}, 4}, {{1,2,0,3}, 4}},
         "threshold: 1000 tuples, 3 orders, arity 4"},
    };

    int warmup = 500;
    int iters = 5000;

    for (auto& cfg : configs) {
        uint32_t* d_data;
        cudaMalloc(&d_data, cfg.arity * cfg.count * sizeof(uint32_t));
        create_data_with_dups(d_data, cfg.count, cfg.arity, cfg.dup_ratio, rng);

        // Allocate per-ordering outputs
        uint32_t* d_out_data[kMaxOrders];
        uint32_t* d_out_counts[kMaxOrders];
        uint32_t h_out_strides[kMaxOrders];
        for (int o = 0; o < cfg.num_orders; o++) {
            cudaMalloc(&d_out_data[o], cfg.arity * cfg.count * sizeof(uint32_t));
            cudaMalloc(&d_out_counts[o], sizeof(uint32_t));
            h_out_strides[o] = cfg.count;
        }

        // Device arrays for kernel params
        uint32_t** d_out_data_arr;
        uint32_t** d_out_count_arr;
        uint32_t* d_strides_arr;
        IndexOrder* d_orders;
        cudaMalloc(&d_out_data_arr, kMaxOrders * sizeof(uint32_t*));
        cudaMalloc(&d_out_count_arr, kMaxOrders * sizeof(uint32_t*));
        cudaMalloc(&d_strides_arr, kMaxOrders * sizeof(uint32_t));
        cudaMalloc(&d_orders, kMaxOrders * sizeof(IndexOrder));
        cudaMemcpy(d_out_data_arr, d_out_data, kMaxOrders * sizeof(uint32_t*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_out_count_arr, d_out_counts, kMaxOrders * sizeof(uint32_t*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_strides_arr, h_out_strides, kMaxOrders * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_orders, cfg.orders, kMaxOrders * sizeof(IndexOrder), cudaMemcpyHostToDevice);

        printf("\n=== %s ===\n", cfg.label);

        // CPU reference
        uint32_t ref_counts[kMaxOrders];
        cpu_multi_sort_dedup(d_data, cfg.count, cfg.count, cfg.arity,
                             cfg.orders, cfg.num_orders, ref_counts);

        // Fused correctness
        for (int o = 0; o < cfg.num_orders; o++)
            cudaMemset(d_out_counts[o], 0, sizeof(uint32_t));

        if (cfg.arity == 3)
            multi_sort_dedup_kernel<3, 2><<<1, kSortBlock>>>(
                d_data, cfg.count, cfg.count,
                d_out_data_arr, d_out_count_arr, d_strides_arr, d_orders);
        else if (cfg.num_orders == 4)
            multi_sort_dedup_kernel<4, 4><<<1, kSortBlock>>>(
                d_data, cfg.count, cfg.count,
                d_out_data_arr, d_out_count_arr, d_strides_arr, d_orders);
        else
            multi_sort_dedup_kernel<4, 3><<<1, kSortBlock>>>(
                d_data, cfg.count, cfg.count,
                d_out_data_arr, d_out_count_arr, d_strides_arr, d_orders);
        cudaDeviceSynchronize();

        bool all_match = true;
        for (int o = 0; o < cfg.num_orders; o++) {
            uint32_t h_count;
            cudaMemcpy(&h_count, d_out_counts[o], sizeof(uint32_t), cudaMemcpyDeviceToHost);
            bool match = h_count == ref_counts[o];
            printf("  Order %d [%d,%d,%d,%d]: ref=%u fused=%u %s\n",
                   o, cfg.orders[o].cols[0], cfg.orders[o].cols[1],
                   cfg.orders[o].cols[2], cfg.orders[o].cols[3],
                   ref_counts[o], h_count, match ? "MATCH" : "MISMATCH!");
            if (!match) all_match = false;
        }

        // Warmup + benchmark fused
        auto run_fused = [&]() {
            for (int o = 0; o < cfg.num_orders; o++)
                cudaMemset(d_out_counts[o], 0, sizeof(uint32_t));
            if (cfg.arity == 3)
                multi_sort_dedup_kernel<3, 2><<<1, kSortBlock>>>(
                    d_data, cfg.count, cfg.count,
                    d_out_data_arr, d_out_count_arr, d_strides_arr, d_orders);
            else if (cfg.num_orders == 4)
                multi_sort_dedup_kernel<4, 4><<<1, kSortBlock>>>(
                    d_data, cfg.count, cfg.count,
                    d_out_data_arr, d_out_count_arr, d_strides_arr, d_orders);
            else
                multi_sort_dedup_kernel<4, 3><<<1, kSortBlock>>>(
                    d_data, cfg.count, cfg.count,
                    d_out_data_arr, d_out_count_arr, d_strides_arr, d_orders);
        };

        for (int i = 0; i < warmup; i++) run_fused();
        cudaDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) run_fused();
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double fused_us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
                          / 1e3 / iters;

        // Thrust: separate sort+dedup per ordering
        uint32_t thrust_counts[kMaxOrders];
        auto run_thrust = [&]() {
            if (cfg.arity == 3)
                run_thrust_multi<3>(d_data, cfg.count, cfg.count,
                                    cfg.orders, cfg.num_orders, d_out_data, thrust_counts);
            else
                run_thrust_multi<4>(d_data, cfg.count, cfg.count,
                                    cfg.orders, cfg.num_orders, d_out_data, thrust_counts);
        };
        for (int i = 0; i < warmup; i++) run_thrust();
        cudaDeviceSynchronize();
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) run_thrust();
        cudaDeviceSynchronize();
        t1 = std::chrono::high_resolution_clock::now();
        double thrust_us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
                           / 1e3 / iters;

        printf("Fused (%d orders): %7.1f us  (1 kernel)\n", cfg.num_orders, fused_us);
        printf("Thrust (%d orders): %7.1f us  (%d × sort+dedup)\n",
               cfg.num_orders, thrust_us, cfg.num_orders);
        printf("Speedup: %.1fx\n", thrust_us / fused_us);

        // Cleanup
        cudaFree(d_data);
        for (int o = 0; o < cfg.num_orders; o++) {
            cudaFree(d_out_data[o]); cudaFree(d_out_counts[o]);
        }
        cudaFree(d_out_data_arr); cudaFree(d_out_count_arr);
        cudaFree(d_strides_arr); cudaFree(d_orders);
    }

    return 0;
}
