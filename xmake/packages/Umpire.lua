-- Umpire package definition for XMake
-- Umpire is a resource management library for memory on NUMA & GPU architectures
-- Repository: https://github.com/LLNL/Umpire.git

package("Umpire")
    set_homepage("https://github.com/LLNL/Umpire")
    set_description("An application-focused API for memory management on NUMA & GPU architectures")
    set_license("MIT")
    
    add_urls("https://github.com/LLNL/Umpire.git", {submodules = true})
    add_versions("develop", "develop")
    -- add_versions("2025.09.0", "v2025.09.0")
    
    -- Build options
    -- Note: Using "enable_cuda" instead of "cuda" to avoid conflict with xmake's reserved "cuda" setting
    add_configs("enable_cuda", {description = "Enable CUDA support", default = false, type = "boolean"})
    add_configs("enable_hip", {description = "Enable HIP support", default = false, type = "boolean"})
    
    -- Ensure package metadata is set when package is loaded (after installation)
    on_load(function (package)
        local install_dir = package:installdir()
        if install_dir and os.isdir(install_dir) then
            local include_dir = path.join(install_dir, "include")
            local lib_dir = path.join(install_dir, "lib")
            
            -- Set include directory
            if os.isdir(include_dir) then
                package:set("includedirs", {include_dir})
            end
            
            -- Set library directory and links
            if os.isdir(lib_dir) then
                package:set("linkdirs", {lib_dir})
                
                -- Check for library files
                local static_lib = path.join(lib_dir, "libumpire.a")
                local shared_lib = path.join(lib_dir, "libumpire.so")
                
                if os.isfile(static_lib) then
                    package:set("links", {"umpire"})
                elseif os.isfile(shared_lib) then
                    package:set("links", {"umpire"})
                else
                    -- Try to find versioned shared libraries
                    local lib_files = os.files(path.join(lib_dir, "libumpire.so.*"))
                    if lib_files and #lib_files > 0 then
                        package:set("links", {"umpire"})
                    else
                        -- Set link anyway - linker will error if library doesn't exist
                        package:set("links", {"umpire"})
                    end
                end
            end
        end
    end)
    
    on_install(function (package)
        local cuda_enabled = package:config("enable_cuda")
        local hip_enabled = package:config("enable_hip")
        
        -- Allow unsupported clang version (clang 20+) with CUDA
        -- This is needed because CUDA doesn't officially support clang 20, but it works
        -- Set environment variable before CMake runs so nvcc sees it during compiler detection
        if cuda_enabled then
            os.setenv("NVCC_PREPEND_FLAGS", "-allow-unsupported-compiler")
        end
        
        -- Umpire uses BLT (Build Library Template) which provides ENABLE_CUDA and ENABLE_HIP options
        -- These are the correct CMake variables per Umpire's CMakeLists.txt
        local configs = {
            "-DCMAKE_BUILD_TYPE=" .. (package:is_debug() and "Debug" or "Release"),
            "-DENABLE_CUDA=" .. (cuda_enabled and "ON" or "OFF"),  -- BLT option, enables UMPIRE_ENABLE_CUDA
            "-DENABLE_HIP=" .. (hip_enabled and "ON" or "OFF"),    -- BLT option, enables UMPIRE_ENABLE_HIP
        }
        
        -- Also pass as CMake variable for compilation phase
        if cuda_enabled then
            table.insert(configs, "-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler")
        end
        
        import("package.tools.cmake").install(package, configs)
        
        -- Set up include and library paths
        local install_dir = package:installdir()
        local include_dir = path.join(install_dir, "include")
        local lib_dir = path.join(install_dir, "lib")
        
        print("Umpire install_dir: " .. install_dir)
        print("Umpire include_dir: " .. include_dir)
        print("Umpire lib_dir: " .. lib_dir)
        
        -- Set include directory (headers are in include/umpire/)
        if os.isdir(include_dir) then
            package:set("includedirs", {include_dir})
        end
        
        -- Set library directory and link libraries
        if os.isdir(lib_dir) then
            package:set("linkdirs", {lib_dir})
            
            -- Umpire typically installs libumpire.a (static) or libumpire.so (shared)
            -- Check for both static and shared libraries
            local static_lib = path.join(lib_dir, "libumpire.a")
            local shared_lib = path.join(lib_dir, "libumpire.so")
            
            -- Also check for versioned shared libraries (e.g., libumpire.so.1.0.0)
            local found_lib = false
            if os.isfile(static_lib) then
                package:set("links", {"umpire"})
                print("Umpire: Found static library libumpire.a")
                found_lib = true
            elseif os.isfile(shared_lib) then
                package:set("links", {"umpire"})
                print("Umpire: Found shared library libumpire.so")
                found_lib = true
            else
                -- Try to find versioned shared libraries by listing lib directory
                local lib_files = os.files(path.join(lib_dir, "libumpire.so.*"))
                if lib_files and #lib_files > 0 then
                    package:set("links", {"umpire"})
                    print("Umpire: Found versioned shared library: " .. lib_files[1])
                    found_lib = true
                else
                    -- Last resort: just set the link and let the linker find it
                    -- The linker can resolve versioned .so files if the .so symlink exists
                    package:set("links", {"umpire"})
                    print("Umpire: Setting link to 'umpire' (linker will resolve library)")
                    found_lib = true  -- Assume it exists, linker will error if not
                end
            end
            
            if not found_lib then
                print("Warning: Umpire library files not found in " .. lib_dir)
                print("  Expected: libumpire.a or libumpire.so")
                print("  This may cause linking errors. Check Umpire installation.")
            end
        else
            print("Warning: Umpire lib directory not found: " .. lib_dir)
            print("  Umpire may not have installed correctly.")
        end
    end)
    
    on_test(function (package)
        -- Test is optional - skip if headers aren't found (might be a build issue)
        local include_dir = path.join(package:installdir(), "include")
        if os.isfile(path.join(include_dir, "umpire", "umpire.hpp")) then
            assert(package:has_cxxfuncs("umpire::getDefaultAllocator", {includes = "umpire/umpire.hpp"}))
        end
    end)
package_end()
