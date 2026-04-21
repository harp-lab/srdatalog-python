package("spdlog")
    set_homepage("https://github.com/gabime/spdlog")
    set_description("Fast C++ logging library.")
    set_license("MIT")

    add_urls("https://github.com/gabime/spdlog/archive/v1.12.0.tar.gz",
             "https://github.com/gabime/spdlog.git")
    add_versions("v1.12.0", "6174bf888528740d5830e2995e50400c4069752b1f8d4885820ee70ce6174a70")

    add_deps("cmake")

    on_install(function (package)
        local configs = {"-DSPDLOG_BUILD_TESTS=OFF", "-DSPDLOG_BUILD_EXAMPLE=OFF"}
        if package:config("shared") then
            table.insert(configs, "-DSPDLOG_BUILD_SHARED=ON")
        else
            table.insert(configs, "-DSPDLOG_BUILD_SHARED=OFF")
        end
        import("package.tools.cmake").install(package, configs)
    end)

    on_test(function (package)
        assert(package:check_cxxsnippets("spdlog::info(\"hello\");", {configs = {languages = "c++11"}, includes = "spdlog/spdlog.h"}))
    end)
package_end()
