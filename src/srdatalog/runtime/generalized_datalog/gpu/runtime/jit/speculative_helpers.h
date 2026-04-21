#pragma once
/**
 * @file speculative_helpers.h
 * @brief Device-side helpers for speculative (sync-free) join execution.
 *
 * claim_region_kernel: after per-rule prefix scan, atomically claims a
 * contiguous write region in the shared output buffer.
 * Sets old_size for the materialize kernel and checks capacity.
 */

#include <cstdint>

namespace SRDatalog::GPU::JIT {

/// Per-destination state for speculative execution (device-resident)
struct DestState {
  uint32_t write_pos;  // atomically incremented by each rule
  uint32_t capacity;   // pre-reserved buffer capacity
  uint32_t overflow;   // 0 = OK, 1 = overflow detected
};

/// Initialize DestState on device — no CPU-GPU sync needed.
/// Launch as <<<1,1,0,stream>>> on any stream.
inline __global__ void init_dest_state_kernel(DestState* __restrict__ dest_state,
                                              uint32_t write_pos, uint32_t capacity) {
  dest_state->write_pos = write_pos;
  dest_state->capacity = capacity;
  dest_state->overflow = 0;
}

/// Claim a contiguous region in the output buffer.
/// Reads the scan total from thread_counts[num_threads],
/// atomically advances dest_state->write_pos, and writes
/// the claimed base offset to *out_old_size for the materialize kernel.
///
/// If the claimed region exceeds capacity, sets overflow flag.
inline __global__ void claim_region_kernel(
    const uint32_t* __restrict__ thread_counts,  // prefix-scanned
    uint32_t num_threads,                        // total at thread_counts[num_threads]
    DestState* __restrict__ dest_state,          // shared per-dest state
    uint32_t* __restrict__ out_old_size          // output: base offset for materialize
) {
  uint32_t my_total = thread_counts[num_threads];
  if (my_total == 0) {
    *out_old_size = 0;
    return;
  }
  uint32_t claimed_base = atomicAdd(&dest_state->write_pos, my_total);
  *out_old_size = claimed_base;
  if (claimed_base + my_total > dest_state->capacity) {
    atomicExch(&dest_state->overflow, 1u);
  }
}

}  // namespace SRDatalog::GPU::JIT
