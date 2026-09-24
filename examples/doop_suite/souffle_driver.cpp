// Link generated Souffle C++ with -D__EMBEDDED_SOUFFLE__.
#include <souffle/SouffleInterface.h>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Clock = std::chrono::steady_clock;

static double seconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double>(end - begin).count();
}

int main(int argc, char** argv) {
    try {
        if (argc != 7) {
            std::cerr << "usage: souffle_reference PROGRAM FACTS OUTPUT THREADS REPORT none|tsv\n";
            return 2;
        }
        const std::string program_name = argv[1];
        const std::filesystem::path facts = argv[2], output = argv[3], report_path = argv[5];
        const std::string format = argv[6];
        if (format != "none" && format != "tsv") {
            throw std::runtime_error("Output format must be none or tsv");
        }
        std::size_t consumed = 0;
        const int threads = std::stoi(argv[4], &consumed);
        if (threads < 1 || consumed != std::string(argv[4]).size()) {
            throw std::runtime_error("threads must be a positive integer");
        }
        if (std::filesystem::exists(report_path)) {
            throw std::runtime_error("Refusing to replace an existing timing report");
        }
        if (format == "tsv" && !std::filesystem::create_directory(output)) {
            throw std::runtime_error("Refusing to replace an existing tuple directory");
        }

        const auto begin = Clock::now();
        std::unique_ptr<souffle::SouffleProgram> program(souffle::ProgramFactory::newInstance(program_name));
        if (!program) {
            throw std::runtime_error("Unknown generated program factory: " + program_name);
        }
        program->setNumThreads(static_cast<std::size_t>(threads));
        const auto instantiated = Clock::now();
        program->loadAll(facts.string());
        const auto loaded = Clock::now();
        // No input/output or intermediate pruning inside the measured fixedpoint.
        // Completion of this synchronous call includes all OpenMP work.
        program->runAll("", "", false, false);
        const auto fixedpoint = Clock::now();

        std::vector<std::pair<std::string, std::size_t>> counts;
        for (auto* relation : program->getAllRelations()) {
            counts.emplace_back(relation->getName(), relation->size());
        }
        const auto counted = Clock::now();
        std::vector<std::string> exported;
        if (format == "tsv") {
            // Every canonical IDB is marked as an output in the translated source.
            // tuple[column] uses logical declaration order, not physical index order.
            for (auto* relation : program->getOutputRelations()) {
                const auto name = relation->getName();
                std::ofstream stream;
                stream.exceptions(std::ios::badbit | std::ios::failbit);
                stream.open(output / (name + ".tsv"));
                for (const auto& tuple : *relation) {
                    for (std::size_t column = 0; column < relation->getArity(); ++column) {
                        const auto value = tuple[column];
                        if (value < std::numeric_limits<std::int32_t>::min()
                                || value > std::numeric_limits<std::int32_t>::max()) {
                            throw std::runtime_error("Tuple outside the common signed int32 domain: " + name);
                        }
                        if (column) stream << '\t';
                        stream << value;
                    }
                    stream << '\n';
                }
                stream.close();
                exported.push_back(name);
            }
        }
        const auto exported_at = Clock::now();
        std::ofstream report;
        report.exceptions(std::ios::badbit | std::ios::failbit);
        report.open(report_path);
        report << std::setprecision(12)
               << "{\"schema_version\":1,\"threads\":" << threads
               << ",\"instantiate_seconds\":" << seconds(begin, instantiated)
               << ",\"load_seconds\":" << seconds(instantiated, loaded)
               << ",\"run_seconds\":" << seconds(loaded, fixedpoint)
               << ",\"count_seconds\":" << seconds(fixedpoint, counted)
               << ",\"export_seconds\":" << seconds(counted, exported_at)
               << ",\"relation_counts\":{";
        bool first = true;
        for (const auto& item : counts) {
            if (!first) report << ',';
            first = false;
            report << '"' << item.first << "\":" << item.second;
        }
        report << "},\"exported_relations\":[";
        first = true;
        for (const auto& name : exported) {
            if (!first) report << ',';
            first = false;
            report << '"' << name << '"';
        }
        report << "]}\n";
        report.close();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CPU reference error: " << error.what() << '\n';
        return 1;
    }
}
