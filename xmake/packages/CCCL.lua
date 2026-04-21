package("CCCL")
    set_homepage("https://github.com/NVIDIA/cccl")
    set_description("CUDA C++ Core Libraries")
    set_license("Apache-2.0")
    
    add_urls("https://github.com/NVIDIA/cccl.git", {submodules = true})
    add_versions("v3.1.3", "v3.1.3")
    set_kind("library", {headeronly = true})
    
    on_install(function (package)
        print("Debugging CCCL structure:")
        os.exec("find . -maxdepth 3")
        -- os.cp("cub/include/cub", package:installdir("include"))
        -- os.cp("thrust/include/thrust", package:installdir("include"))
        -- os.cp("libcudacxx/include/cuda", package:installdir("include"))
        -- os.cp("libcudacxx/include/nv", package:installdir("include")) 
    end)
    
    on_test(function (package)
        -- Test if headers are available
        assert(package:check_cxxsnippets("#include <cuda/std/type_traits>", {configs = {languages = "c++17"}}))
    end)
package_end()
