#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

// Stub implementation for missing hsa_amd_memory_get_preferred_copy_engine
// This function was introduced in ROCm 7.0.0 but may not be available in all installations
hsa_status_t hsa_amd_memory_get_preferred_copy_engine(hsa_agent_t dst_agent, 
                                                       hsa_agent_t src_agent,
                                                       uint32_t* recommended_ids_mask) {
    // Return a default value - use all available copy engines
    if (recommended_ids_mask) {
        *recommended_ids_mask = 0xFFFFFFFF;  // All engines
    }
    return HSA_STATUS_SUCCESS;
}
