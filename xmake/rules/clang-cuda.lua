-- Clang-CUDA rule: configures CUDA targets using clang-cuda toolchain
-- Usage: add_rules("clang-cuda.target") with set_toolchains("clang-cuda")

rule("clang-cuda.target")
    on_config(function (target)
        local toolchain = target:toolchain("clang-cuda")
        if toolchain then
            local cuda_lib_path = toolchain:get("cuda_lib_path")
            if cuda_lib_path and os.isdir(cuda_lib_path) then
                target:set("rpathdirs", cuda_lib_path)
            end
        end
    end)
    
    after_load(function (target)
        local toolchain = target:toolchain("clang-cuda")
        if not toolchain then
            return
        end
        
        -- CUDA device code compilation needs explicit -I flags for package includes
        import("core.package.package")
        
        local common_packages = {"RMM", "spdlog", "highway", "boost", "benchmark", "tbb", "mimalloc"}
        for _, pkgname in ipairs(common_packages) do
            local pkg = target:pkg(pkgname)
            if pkg then
                local includedirs = pkg:get("includedirs")
                if includedirs then
                    for _, incdir in ipairs(includedirs) do
                        if type(incdir) == "string" and os.isdir(incdir) then
                            target:add("includedirs", incdir, {force = true})
                            target:add("cxflags", "-I" .. incdir, {force = true})
                        end
                    end
                else
                    local installdir = pkg:installdir()
                    if installdir and os.isdir(installdir) then
                        local include_dir = path.join(installdir, "include")
                        if os.isdir(include_dir) then
                            target:add("includedirs", include_dir, {force = true})
                            target:add("cxflags", "-I" .. include_dir, {force = true})
                        end
                    end
                end
                
                -- For nvbench, also ensure linking is set up (library + CUPTI)
                -- if pkgname == "nvbench" then
                --     local linkdirs = pkg:get("linkdirs")
                --     if linkdirs then
                --         for _, linkdir in ipairs(linkdirs) do
                --             if type(linkdir) == "string" and os.isdir(linkdir) then
                --                 target:add("linkdirs", linkdir, {force = true})
                --             end
                --         end
                --     else
                --         local installdir = pkg:installdir()
                --         if installdir and os.isdir(installdir) then
                --             local lib_dir = path.join(installdir, "lib")
                --             if os.isdir(lib_dir) then
                --                 target:add("linkdirs", lib_dir, {force = true})
                --             end
                --         end
                --     end
                    
                --     local links = pkg:get("links")
                --     if links then
                --         for _, link in ipairs(links) do
                --             target:add("links", link, {force = true})
                --         end
                --     else
                --         -- Fallback: try to link nvbench library
                --         target:add("links", "nvbench", {force = true})
                --     end
                    
                --     -- Add CUDA driver API and NVML (syslinks from package)
                --     local syslinks = pkg:get("syslinks")
                --     if syslinks then
                --         for _, syslink in ipairs(syslinks) do
                --             target:add("syslinks", syslink, {force = true})
                --         end
                --     else
                --         -- Fallback: add common syslinks
                --         target:add("syslinks", "cuda", "nvidia-ml", {force = true})
                --     end
                    
                --     -- Also add NVML library path and link flags from package
                --     local ldflags = pkg:get("ldflags")
                --     if ldflags then
                --         for _, ldflag in ipairs(ldflags) do
                --             target:add("ldflags", ldflag, {force = true})
                --         end
                --     end
                    
                --     -- Also ensure linkdirs from package are applied
                --     local linkdirs = pkg:get("linkdirs")
                --     if linkdirs then
                --         for _, linkdir in ipairs(linkdirs) do
                --             if type(linkdir) == "string" and os.isdir(linkdir) then
                --                 target:add("linkdirs", linkdir, {force = true})
                --             end
                --         end
                --     end
                    
                --     -- Fallback: add NVML from NVHPC stubs if not already added
                --     local cuda_path = toolchain:get("cuda_path")
                --     if cuda_path then
                --         local nvml_stubs_path = path.join(cuda_path, "targets", "x86_64-linux", "lib", "stubs")
                --         if os.isdir(nvml_stubs_path) then
                --             target:add("linkdirs", nvml_stubs_path, {force = true})
                --             target:add("ldflags", "-L" .. nvml_stubs_path, "-lnvidia-ml", {force = true})
                --         end
                --     end
                    
                --     -- Add CUPTI libraries (from package's ldflags)
                --     local ldflags = pkg:get("ldflags")
                --     if ldflags then
                --         for _, ldflag in ipairs(ldflags) do
                --             target:add("ldflags", ldflag, {force = true})
                --         end
                --     else
                --         -- Fallback: add CUPTI libraries manually
                --         local cupti_lib = "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/extras/CUPTI/lib64"
                --         if os.isdir(cupti_lib) then
                --             target:add("linkdirs", cupti_lib, {force = true})
                --             target:add("ldflags", "-L" .. cupti_lib, {force = true})
                --             target:add("ldflags", "-lcupti", "-lnvperf_host", "-lnvperf_target", {force = true})
                --         end
                --     end
                -- end
            end
        end
        
        -- Manual fallback for RMM if package wasn't found via xmake
        -- Try to find RMM in common installation locations
        if not target:pkg("RMM") then
            local rmm_paths = {
                path.join(os.getenv("HOME") or "", ".xmake/packages/r/rmm/v24.12.01"),
                path.join(os.getenv("HOME") or "", ".xmake/packages/r/rmm/v24.02.00"),
                path.join(os.getenv("HOME") or "", ".xmake/packages/r/rmm/dev"),
            }
            for _, rmm_base in ipairs(rmm_paths) do
                if os.isdir(rmm_base) then
                    -- Find the actual install directory (hash-named subdirectory)
                    local subdirs = os.dirs(path.join(rmm_base, "*"))
                    if subdirs and #subdirs > 0 then
                        local rmm_include = path.join(subdirs[1], "include")
                        if os.isdir(rmm_include) then
                            target:add("includedirs", rmm_include, {force = true})
                            target:add("cxflags", "-I" .. rmm_include, {force = true})
                            -- print("RMM: Found and added include path: " .. rmm_include)
                            break
                        end
                    end
                end
            end
        end
        
        -- Manual fallback for spdlog if package wasn't found via xmake
        -- spdlog is a dependency of RMM
        if not target:pkg("spdlog") then
            local spdlog_paths = {
                path.join(os.getenv("HOME") or "", ".xmake/packages/s/spdlog/v1.16.0"),
            }
            for _, spdlog_base in ipairs(spdlog_paths) do
                if os.isdir(spdlog_base) then
                    local subdirs = os.dirs(path.join(spdlog_base, "*"))
                    if subdirs and #subdirs > 0 then
                        local spdlog_include = path.join(subdirs[1], "include")
                        if os.isdir(spdlog_include) then
                            target:add("includedirs", spdlog_include, {force = true})
                            target:add("cxflags", "-I" .. spdlog_include, {force = true})
                            -- print("spdlog: Found and added include path: " .. spdlog_include)
                            break
                        end
                    end
                end
            end
        end
    end)
    
    before_link(function (target)
        local toolchain = target:toolchain("clang-cuda")
        if toolchain then
            local cuda_lib_path = toolchain:get("cuda_lib_path")
            if cuda_lib_path and os.isdir(cuda_lib_path) then
                local lib_path_env = os.getenv("LIBRARY_PATH") or ""
                if not lib_path_env:match(cuda_lib_path) then
                    local new_lib_path = cuda_lib_path .. (lib_path_env ~= "" and (":" .. lib_path_env) or "")
                    os.setenv("LIBRARY_PATH", new_lib_path)
                end
                target:add("ldflags", "-L" .. cuda_lib_path)
            end
        end
    end)
rule_end()

