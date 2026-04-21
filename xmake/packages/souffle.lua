package("souffle")
    set_homepage("https://github.com/souffle-lang/souffle")
    set_description("Souffle is a translator of Datalog programs into C++ code.")
    set_license("UPL-1.0")

    add_urls("https://github.com/souffle-lang/souffle.git")
    -- We force master as requested and to avoid build issues in 2.x
    add_versions("master", "master")
    add_deps("cmake")
    add_deps("m4", "bison", "flex")
    add_deps("libffi", "zlib", "sqlite3", "ncurses")

    on_install("linux", "macosx", function (package)
        local configs = {
            "-DSOUFFLE_DOMAIN_64BIT=ON",
            "-DSOUFFLE_GIT=OFF",
            "-DSOUFFLE_WERROR=OFF",
            "-DSOUFFLE_ENABLE_TESTING=OFF",
            "-DBUILD_TESTING=OFF",
            "-DCMAKE_COMPILE_WARNING_AS_ERROR=OFF",
            "-DCMAKE_CXX_STANDARD=20",
            "-DCMAKE_CXX_STANDARD_REQUIRED=ON"
        }
        
        -- Fix tcb namespace error in C++20/23 by forcing bundled span to be used
        io.replace("src/include/souffle/utility/span.h", "#if __cplusplus >= 202000L", "#if 0", {plain = true})
        
        -- Fix std::atomic assignment error in Index.h
        io.replace("src/interpreter/Index.h", "data = src.data;", "data.store(src.data.load());", {plain = true})

        -- Fix memory_order_relaxed error in Brie.h
        io.replace("src/include/souffle/datastructure/Brie.h", "std::memory_order::memory_order_relaxed", "std::memory_order_relaxed", {plain = true})

        -- Fix tcb namespace error in WriteStream.h by using souffle::span (aliased to std::span in C++20)
        io.replace("src/include/souffle/io/WriteStream.h", "using tcb::make_span;", "using namespace souffle;", {plain = true})
        io.replace("src/include/souffle/io/WriteStream.h", "make_span(tuple)", "span(tuple)", {plain = true})

        -- Clang 15+ is strict about unqualified std::forward/std::move
        -- and C++20/23 features used in stricter environments.
        local cxflags = "-Wno-error -Wno-unqualified-std-cast-call -Wno-vla-cxx-extension"
        table.insert(configs, "-DCMAKE_CXX_FLAGS=" .. cxflags)

        import("package.tools.cmake").install(package, configs)
    end)

    on_test(function (package)
        -- Just check if souffle binary exists
        os.run("souffle --version")
    end)
package_end()
