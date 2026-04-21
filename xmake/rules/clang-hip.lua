-- Clang-HIP rule: configures HIP targets using clang-hip toolchain
-- Usage: add_rules("clang-hip.target") with set_toolchains("clang-hip")

rule("clang-hip.target")
    on_config(function (target)
        local toolchain = target:toolchain("clang-hip")
        if toolchain then
            local rocm_lib_path = toolchain:get("rocm_lib_path")
            if rocm_lib_path and os.isdir(rocm_lib_path) then
                target:set("rpathdirs", rocm_lib_path)
            end
            -- Also add ROCm 7.1.1 lib path to fix HSA runtime version mismatch
            -- The symlink /opt/rocm/lib/libhsa-runtime64.so.1 points to 6.4.2, but we need 7.1.1
            -- Add to both rpath and LD_LIBRARY_PATH so it's found at runtime
            local rocm_711_lib_path = "/opt/rocm-7.1.1/lib"
            if os.isdir(rocm_711_lib_path) then
                -- Add to rpathdirs
                target:add("rpathdirs", rocm_711_lib_path, {force = true})
                -- Also add to LD_LIBRARY_PATH for runtime (prepend so it's searched first)
                local existing_ld_path = os.getenv("LD_LIBRARY_PATH") or ""
                local new_ld_path = rocm_711_lib_path .. (existing_ld_path ~= "" and (":" .. existing_ld_path) or "")
                target:add("runenvs", "LD_LIBRARY_PATH", new_ld_path)
            end
            
            -- Workaround for ROCm 7.1.1 library version mismatch:
            -- hsa_amd_memory_get_preferred_copy_engine symbol is missing from HSA runtime
            -- Build and use a stub library that provides this symbol via LD_PRELOAD
            local project_dir = os.projectdir()
            local hsa_stub_script = path.join(project_dir, "generalized_datalog", "gpu", "build_hsa_stub.sh")
            local hsa_stub_so = path.join(project_dir, "build", "libhsa_stub.so")
            if os.isfile(hsa_stub_script) and not os.isfile(hsa_stub_so) then
                -- Build the stub library if script exists and library doesn't
                os.execv("sh", {"-c", "bash " .. hsa_stub_script .. " 2>&1"})
            end
            if os.isfile(hsa_stub_so) then
                local existing_preload = os.getenv("LD_PRELOAD") or ""
                local new_preload = hsa_stub_so .. (existing_preload ~= "" and (":" .. existing_preload) or "")
                target:add("runenvs", "LD_PRELOAD", new_preload)
            end
        end
    end)
    
    after_load(function (target)
        local toolchain = target:toolchain("clang-hip")
        if not toolchain then
            return
        end
        
        -- HIP device code compilation needs explicit -I flags for package includes
        import("core.package.package")
        
        -- Common packages that need to be accessible in HIP device code
        -- Note: hipMM replaces RMM for HIP builds
        local common_packages = {"hipMM", "spdlog", "highway", "boost", "benchmark", "tbb", "mimalloc"}
        
        -- Special handling for hipMM: add libhipcxx include path (provides cuda/stream_ref)
        -- and link bundled libraries (spdlog, fmt) since hipMM is header-only
        local hipmm_pkg = target:pkg("hipMM")
        if hipmm_pkg then
            local includedirs = hipmm_pkg:get("includedirs")
            local hipmm_include_path = nil
            local hipmm_installdir = hipmm_pkg:installdir()
            
            if includedirs and #includedirs > 0 then
                hipmm_include_path = includedirs[1]
            elseif hipmm_installdir and os.isdir(hipmm_installdir) then
                hipmm_include_path = path.join(hipmm_installdir, "include")
            end
            
            if hipmm_include_path and os.isdir(hipmm_include_path) then
                local libhipcxx_path = path.join(hipmm_include_path, "rapids", "libhipcxx")
                if os.isdir(libhipcxx_path) then
                    target:add("includedirs", libhipcxx_path, {force = true})
                    target:add("cxflags", "-I" .. libhipcxx_path, {force = true})
                    -- print("clang-hip: Added libhipcxx include path (from hipMM): " .. libhipcxx_path)
                end
            end
            
            -- Link bundled libraries (spdlog and fmt) from hipMM's lib directory
            -- hipMM is header-only but bundles these libraries that RMM code needs
            if hipmm_installdir and os.isdir(hipmm_installdir) then
                local hipmm_lib_dir = path.join(hipmm_installdir, "lib")
                if os.isdir(hipmm_lib_dir) then
                    -- Add hipMM lib directory to RPATH so bundled libraries (spdlog, fmt) can be found at runtime
                    target:add("rpathdirs", hipmm_lib_dir, {force = true})
                    target:add("ldflags", "-Wl,-rpath," .. hipmm_lib_dir, {force = true})
                    target:add("linkdirs", hipmm_lib_dir)
                    -- Check for spdlog library
                    local spdlog_lib = path.join(hipmm_lib_dir, "libspdlog.so")
                    local spdlog_lib_static = path.join(hipmm_lib_dir, "libspdlog.a")
                    if os.isfile(spdlog_lib) or os.isfile(spdlog_lib_static) then
                        target:add("links", "spdlog")
                        -- print("clang-hip: Linking bundled spdlog from hipMM: " .. hipmm_lib_dir)
                    end
                    -- Check for fmt library
                    local fmt_lib = path.join(hipmm_lib_dir, "libfmt.so")
                    local fmt_lib_static = path.join(hipmm_lib_dir, "libfmt.a")
                    if os.isfile(fmt_lib) or os.isfile(fmt_lib_static) then
                        target:add("links", "fmt")
                        -- print("clang-hip: Linking bundled fmt from hipMM: " .. hipmm_lib_dir)
                    end
                end
            end
        end
        
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
            end
        end
        
        -- Manual fallback for hipMM if package wasn't found via xmake
        if not target:pkg("hipMM") then
            local hipmm_paths = {
                path.join(os.getenv("HOME") or "", ".xmake/packages/h/hipmm/v3.0.0"),
                path.join(os.getenv("HOME") or "", ".xmake/packages/h/hipmm/dev"),
            }
            for _, hipmm_base in ipairs(hipmm_paths) do
                if os.isdir(hipmm_base) then
                    local subdirs = os.dirs(path.join(hipmm_base, "*"))
                    if subdirs and #subdirs > 0 then
                        local hipmm_include = path.join(subdirs[1], "include")
                        if os.isdir(hipmm_include) then
                            target:add("includedirs", hipmm_include, {force = true})
                            target:add("cxflags", "-I" .. hipmm_include, {force = true})
                            break
                        end
                    end
                end
            end
        end
        
        -- Manual fallback for spdlog if package wasn't found via xmake
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
                            break
                        end
                    end
                end
            end
        end
        
        -- Prefer GitHub rocprim headers to ensure namespace visibility
        -- Even though 7.1.1 headers are identical, there may be include order issues
        local project_dir = os.projectdir()
        local deps_rocprim = path.join(project_dir, "deps", "rocprim", "include")
        if os.isdir(deps_rocprim) then
            -- Add GitHub headers FIRST to ensure they're found before system headers
            target:add("includedirs", deps_rocprim, {force = true})
            target:add("cxflags", "-I" .. deps_rocprim, {force = true})
            -- print("clang-hip: Using GitHub rocprim headers (for namespace visibility): " .. deps_rocprim)
        end
        
        -- Add hipCUB backend path for CUB headers (needed for device code)
        local rocm_path = toolchain:get("rocm_path") or "/opt/rocm"
        local hipcub_backend = path.join(rocm_path, "include", "hipcub", "backend")
        if os.isdir(hipcub_backend) then
            target:add("includedirs", hipcub_backend, {force = true})
            target:add("cxflags", "-I" .. hipcub_backend, {force = true})
        end
        
        -- Add GPU shim directory for CUDA compatibility headers (cuda/stream_ref, etc.)
        local project_dir = os.projectdir()
        local gpu_shim_dir = path.join(project_dir, "generalized_datalog", "gpu", "shim")
        if os.isdir(gpu_shim_dir) then
            target:add("includedirs", gpu_shim_dir, {force = true})
            target:add("cxflags", "-I" .. gpu_shim_dir, {force = true})
        end
        
        -- Add HIP/ROCm compatibility defines
        target:add("defines", "__HIP_PLATFORM_AMD__")
        target:add("defines", "LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE")
        
        -- Workaround: hipMM is header-only and doesn't provide RMM logger implementation
        -- Define RMM_LOGGING_LEVEL=0 to disable logging and avoid missing logger symbols
        target:add("defines", "RMM_LOGGING_LEVEL=0")
        
        -- Add RMM logger stub implementation for ROCm builds
        -- This provides stub implementations of missing logger symbols
        local project_dir = os.projectdir()
        local rmm_stub_file = path.join(project_dir, "generalized_datalog", "gpu", "rmm_logger_stub.cpp")
        if os.isfile(rmm_stub_file) then
            target:add("files", rmm_stub_file)
            -- print("clang-hip: Added RMM logger stub implementation: " .. rmm_stub_file)
        end
    end)
    
    before_link(function (target)
        local toolchain = target:toolchain("clang-hip")
        if toolchain then
            local rocm_lib_path = toolchain:get("rocm_lib_path")
            if rocm_lib_path and os.isdir(rocm_lib_path) then
                local lib_path_env = os.getenv("LIBRARY_PATH") or ""
                if not lib_path_env:match(rocm_lib_path) then
                    local new_lib_path = rocm_lib_path .. (lib_path_env ~= "" and (":" .. lib_path_env) or "")
                    os.setenv("LIBRARY_PATH", new_lib_path)
                end
                target:add("ldflags", "-L" .. rocm_lib_path)
            end
            
            -- Workaround: hipMM is header-only and doesn't provide RMM logger implementation
            -- Use linker flags to allow undefined RMM logger symbols (they'll be no-ops at runtime)
            -- This is safe because RMM_LOGGING_LEVEL=0 disables logging anyway
            target:add("ldflags", "-Wl,--allow-shlib-undefined", {force = true})
            target:add("ldflags", "-Wl,--unresolved-symbols=ignore-in-shared-libs", {force = true})
        end
    end)
rule_end()


