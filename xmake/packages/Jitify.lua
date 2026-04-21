package("Jitify")
    set_homepage("https://github.com/NVIDIA/jitify")
    set_description("A single-header C++ library for simplifying the use of CUDA Runtime Compilation (NVRTC)")
    set_license("BSD-3-Clause")
    
    -- Jitify is available on the jitify2 branch
    add_urls("https://github.com/NVIDIA/jitify.git", {submodules = false})
    add_versions("jitify2", "jitify2")
    
    -- Jitify is header-only, but needs -ldl for dynamic library loading
    set_kind("library", {headeronly = true})
    
    on_install(function (package)
        local install_dir = package:installdir()
        local include_dir = path.join(install_dir, "include")
        
        -- Create include directory
        os.mkdir(include_dir)
        
        -- Find jitify2.hpp recursively from current directory
        -- When on_install is called, xmake has already cloned the repo to current dir
        local current_dir = os.curdir()
        local found_file = nil
        
        -- Try to find the file using os.files with recursive pattern
        local all_hpp_files = os.files(path.join(current_dir, "**/jitify2.hpp"))
        if all_hpp_files and #all_hpp_files > 0 then
            found_file = all_hpp_files[1]
        else
            -- Try common locations
            local possible_paths = {
                path.join(current_dir, "jitify2.hpp"),
                path.join(current_dir, "Jitify", "jitify2.hpp"),
                "jitify2.hpp",
                "Jitify/jitify2.hpp",
            }
            for _, possible_path in ipairs(possible_paths) do
                if os.isfile(possible_path) then
                    found_file = possible_path
                    break
                end
            end
        end
        
        if found_file and os.isfile(found_file) then
            os.cp(found_file, include_dir)
            print("Jitify: Copied jitify2.hpp from " .. found_file .. " to " .. include_dir)
        else
            raise("Jitify: jitify2.hpp not found. Searched from: " .. current_dir)
        end
        
        -- Set include directory
        package:set("includedirs", {include_dir})
        
        -- Note: Jitify requires -ldl for dynamic library loading
        -- This should be handled by the target configuration, not here
    end)
    
    on_load(function (package)
        local install_dir = package:installdir()
        if install_dir and os.isdir(install_dir) then
            local include_dir = path.join(install_dir, "include")
            if os.isdir(include_dir) then
                package:set("includedirs", {include_dir})
            end
        end
    end)
    
    on_test(function (package)
        local include_dir = path.join(package:installdir(), "include")
        local header_file = path.join(include_dir, "jitify2.hpp")
        if os.isfile(header_file) then
            -- Header file exists, test passed
            -- Skip compilation test as it requires CUDA/NVRTC which may not be available during package install
            return
        else
            raise("Jitify: jitify2.hpp not found in " .. include_dir)
        end
    end)
package_end()

