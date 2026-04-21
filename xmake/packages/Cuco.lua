package("Cuco")
    set_homepage("https://github.com/rapidsai/cuco")
    set_description("NVIDIA CUDA Concurrent Hash Map")
    set_license("Apache-2.0")
    
    add_urls("https://github.com/NVIDIA/cuCollections.git", {submodules = true})
    add_versions("dev", "dev")
    add_versions("v24.02.00", "dev") -- Map requested version to dev branch if tag missing
    
    add_deps("cmake")
    add_deps("RMM") -- Depends on RMM? Usually yes for allocators
    
    on_install(function (package)
        local configs = {
            "-DBUILD_TESTS=OFF",
            "-DBUILD_BENCHMARKS=OFF",
            "-DBUILD_EXAMPLES=OFF"
        }
        
        -- Find CUDA/Thrust paths
        local cuda_path = os.getenv("ACPP_CUDA_PATH") or 
                          os.getenv("CUDA_PATH") or 
                          os.getenv("CUDA_HOME") or
                          "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9"
        
        local target_path = path.join(cuda_path, "targets", "x86_64-linux")
        
        table.insert(configs, "-DThrust_ROOT=" .. target_path)
        table.insert(configs, "-DCUB_ROOT=" .. target_path)
        
        os.setenv("NVCC_PREPEND_FLAGS", "-allow-unsupported-compiler")
        import("package.tools.cmake").install(package, configs)
    end)
    
    -- on_test removed
package_end()
