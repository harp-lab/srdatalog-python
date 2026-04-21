-- CUBThrust package definition for XMake
-- Auto-discovers CUB/Thrust based on AdaptiveCpp backend configuration
-- Supports both NVIDIA (CCCL) and AMD (ROCm) implementations

package("CUBThrust")
    set_homepage("https://github.com/NVIDIA/cccl (NVIDIA) or https://github.com/ROCmSoftwarePlatform/hipCUB (AMD)")
    set_description("CUB/Thrust library - auto-detected based on GPU vendor")
    
    on_load(function (package)
        -- Determine vendor from AdaptiveCpp targets or environment
        local vendor = nil
        local acpp_targets = os.getenv("ACPP_TARGETS") or ""
        acpp_targets = acpp_targets:lower()
        
        if acpp_targets:match("cuda") then
            vendor = "nvidia"
        elseif acpp_targets:match("hip") or acpp_targets:match("rocm") then
            vendor = "amd"
        end
        
        -- If not determined, try to detect from available tools/paths
        if not vendor then
            -- Try to detect nvcc if available
            local nvcc_found = false
            local find_tool = import("lib.detect.find_tool")
            if find_tool then
                local nvcc = find_tool("nvcc")
                if nvcc then
                    nvcc_found = true
                    vendor = "nvidia"
                end
            end
            
            -- If still not determined, check paths
            if not vendor then
                if os.isdir("/opt/rocm") or os.getenv("ROCM_PATH") then
                    vendor = "amd"
                else
                    -- Check for CUDA installation path (same as used in xmake_benchmarks.lua)
                    local cuda_path = os.getenv("ACPP_CUDA_PATH") or 
                                     os.getenv("CUDA_PATH") or 
                                     os.getenv("CUDA_HOME") or
                                     "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9"
                    if os.isdir(cuda_path) or os.isdir("/opt/nvidia/hpc_sdk") then
                        vendor = "nvidia"
                    elseif os.isdir("/opt/rocm") or os.getenv("ROCM_PATH") then
                        vendor = "amd"
                    else
                        vendor = "nvidia"  -- Default to NVIDIA
                    end
                end
            end
        end
        
        package:set("vendor", vendor)
        
        -- NVIDIA (CCCL) Configuration
        if vendor == "nvidia" then
            -- Use ACPP_CUDA_PATH if available (from xmake_benchmarks.lua or environment)
            -- Default to the same path used in xmake_benchmarks.lua
            local default_cuda_path = "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9"
            local cuda_path = os.getenv("ACPP_CUDA_PATH") or 
                             os.getenv("CUDA_PATH") or 
                             os.getenv("CUDA_HOME") or
                             default_cuda_path
            
            -- Verify the configured path has Thrust headers, otherwise use default
            local configured_target_include = path.join(cuda_path, "targets", "x86_64-linux", "include")
            local has_thrust = os.isfile(path.join(configured_target_include, "thrust", "version.h")) or
                               os.isfile(path.join(configured_target_include, "cccl", "thrust", "version.h"))
            
            if not has_thrust and cuda_path ~= default_cuda_path then
                -- Try default path
                local default_target_include = path.join(default_cuda_path, "targets", "x86_64-linux", "include")
                local default_has_thrust = os.isfile(path.join(default_target_include, "thrust", "version.h")) or
                                           os.isfile(path.join(default_target_include, "cccl", "thrust", "version.h"))
                if default_has_thrust then
                    cuda_path = default_cuda_path
                    print("CUBThrust: Configured CUDA path missing Thrust headers, using default: " .. cuda_path)
                end
            end
                        
            -- Check for CUB/Thrust in CUDA toolkit
            -- Try multiple possible locations including CCCL structure
            -- For CCCL (newer CUDA), headers are in targets/x86_64-linux/include/cccl/
            local search_paths = {}
            -- Try the configured CUDA path first (highest priority)
            table.insert(search_paths, path.join(cuda_path, "targets", "x86_64-linux", "include"))
            table.insert(search_paths, path.join(cuda_path, "include"))
            -- Add fallback paths
            table.insert(search_paths, "/usr/local/cuda/targets/x86_64-linux/include")
            table.insert(search_paths, "/usr/local/cuda/include")
            table.insert(search_paths, "/opt/cuda/include")
            
            local cub_include = nil
            local thrust_include = nil
            
            -- First, try the configured CUDA path specifically (highest priority)
            local configured_target_include = path.join(cuda_path, "targets", "x86_64-linux", "include")
            if os.isdir(configured_target_include) then
                -- Check for CUB
                if os.isfile(path.join(configured_target_include, "cub", "cub.cuh")) then
                    cub_include = configured_target_include
                elseif os.isfile(path.join(configured_target_include, "cccl", "cub", "cub.cuh")) then
                    cub_include = configured_target_include
                end
                -- Check for Thrust
                if os.isfile(path.join(configured_target_include, "thrust", "version.h")) then
                    thrust_include = configured_target_include
                elseif os.isfile(path.join(configured_target_include, "cccl", "thrust", "version.h")) then
                    thrust_include = configured_target_include
                end
            end
            
            -- If not found in configured path, search other paths
            if not cub_include or not thrust_include then
                for _, p in ipairs(search_paths) do
                    if os.isdir(p) then
                        -- Check for CUB (can be in cub/ or cccl/cub/)
                        if not cub_include then
                            if os.isfile(path.join(p, "cub", "cub.cuh")) then
                                cub_include = p
                            elseif os.isfile(path.join(p, "cccl", "cub", "cub.cuh")) then
                                cub_include = p
                            end
                        end
                        -- Check for Thrust (can be in thrust/ or cccl/thrust/)
                        if not thrust_include then
                            if os.isfile(path.join(p, "thrust", "version.h")) then
                                thrust_include = p
                            elseif os.isfile(path.join(p, "cccl", "thrust", "version.h")) then
                                thrust_include = p
                            end
                        end
                        -- Stop searching if both found
                        if cub_include and thrust_include then
                            break
                        end
                    end
                end
            end
            
            if cub_include and thrust_include then
                package:set("cub_include", cub_include)
                package:set("thrust_include", thrust_include)
                package:set("cuda_path", cuda_path)
            else
                -- Debug: print what we found
                print("CUBThrust: NVIDIA vendor detected but headers not found")
                print("  CUDA path checked: " .. cuda_path)
                print("  CUB found: " .. tostring(cub_include or "no"))
                print("  Thrust found: " .. tostring(thrust_include or "no"))
            end
        end
        
        -- AMD (ROCm) Configuration
        if vendor == "amd" then
            local rocm_path = os.getenv("ROCM_PATH") or "/opt/rocm"
            
            local rocm_include_paths = {
                path.join(rocm_path, "include"),
                path.join(rocm_path, "hipcub", "include"),
                path.join(rocm_path, "hipthrust", "include"),
                "/opt/rocm/include",
                "/usr/local/rocm/include"
            }
            
            local cub_include = nil
            local thrust_include = nil
            
            for _, p in ipairs(rocm_include_paths) do
                if os.isdir(p) then
                    if os.isfile(path.join(p, "hipcub", "hipcub.hpp")) then
                        cub_include = p
                    end
                    if os.isfile(path.join(p, "hipthrust", "hipthrust.h")) then
                        thrust_include = p
                    end
                end
            end
            
            if cub_include and thrust_include then
                package:set("cub_include", cub_include)
                package:set("thrust_include", thrust_include)
                package:set("rocm_path", rocm_path)
            end
        end
    end)
    
    on_fetch(function (package)
        local vendor = package:get("vendor")
        local cub_include = package:get("cub_include")
        local thrust_include = package:get("thrust_include")
        
        -- Always return a result if vendor is detected
        local result = {}
        
        -- Set includes if headers were found
        if cub_include and thrust_include then
            -- Remove duplicates if both point to the same directory
            local includedirs = {}
            if cub_include == thrust_include then
                table.insert(includedirs, cub_include)
            else
                table.insert(includedirs, cub_include)
                table.insert(includedirs, thrust_include)
            end
            result.includedirs = includedirs
            
            -- Add vendor-specific defines
            if vendor == "nvidia" then
                result.defines = {"CUBTHRUST_VENDOR_NVIDIA"}
                print("CUBThrust: Found NVIDIA headers, setting CUBTHRUST_VENDOR_NVIDIA")
                print("  CUB include: " .. cub_include)
                print("  Thrust include: " .. thrust_include)
            elseif vendor == "amd" then
                result.defines = {"CUBTHRUST_VENDOR_AMD"}
                print("CUBThrust: Found AMD headers, setting CUBTHRUST_VENDOR_AMD")
            end
        else
            -- Headers not found, but still set the define based on vendor
            -- This allows the code to compile (though Thrust won't work)
            result.includedirs = {}
            if vendor == "nvidia" then
                result.defines = {"CUBTHRUST_VENDOR_NVIDIA"}
                print("CUBThrust: Setting CUBTHRUST_VENDOR_NVIDIA (headers not found)")
                print("  This allows compilation but Thrust APIs will not work")
            elseif vendor == "amd" then
                result.defines = {"CUBTHRUST_VENDOR_AMD"}
                print("CUBThrust: Setting CUBTHRUST_VENDOR_AMD (headers not found)")
            else
                -- No vendor detected, return nil to skip package
                return nil
            end
        end
        
        return result
    end)

package_end()

