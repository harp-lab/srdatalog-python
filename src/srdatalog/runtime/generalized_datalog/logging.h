#pragma once

// Logging wrapper that can be disabled at compile time
// Usage:
//   LOG_TRACE << "message";
//   LOG_DEBUG << "message";
//   LOG_INFO << "message";
//   LOG_WARNING << "message";
//   LOG_ERROR << "message";
//   LOG_FATAL << "message";
//
// Diagnostic logging (for detailed debugging):
//   LOG_DIAG << "detailed diagnostic info";
//   if (LOG_DIAG_ENABLED) { /* expensive diagnostic code */ }
#include <cstdlib>
#include <iostream>
#include <string>

#ifdef ENABLE_LOGGING
// Include Boost headers from separate header to ensure they're at global scope
// This prevents Boost's internal code from looking in SRDatalog::boost namespace
#include "logging_boost.h"

namespace SRDatalog::Logging {

// Initialize logging system (call once at program start)
inline void init_logging(bool to_console = true, bool to_file = false,
                         const std::string& log_file = "srdatalog.log") {
  namespace logging = boost::log;
  namespace keywords = boost::log::keywords;

  // Add common attributes (timestamp, thread ID, etc.)
  logging::add_common_attributes();

  // Console sink
  if (to_console) {
    logging::add_console_log(
        std::clog, keywords::format = "[%TimeStamp%] [%Severity%] %Message%",
        keywords::filter = logging::trivial::severity >= logging::trivial::info);
  }

  // File sink
  if (to_file) {
    logging::add_file_log(keywords::file_name = log_file,
                          keywords::format = "[%TimeStamp%] [%ThreadID%] [%Severity%] %Message%",
                          keywords::auto_flush = true);
  }

  // Set default log level
  logging::core::get()->set_filter(logging::trivial::severity >= logging::trivial::trace);
}

// Convenience macros
#define LOG_TRACE BOOST_LOG_TRIVIAL(trace)
#define LOG_DEBUG BOOST_LOG_TRIVIAL(debug)
#define LOG_INFO BOOST_LOG_TRIVIAL(info)
#define LOG_WARNING BOOST_LOG_TRIVIAL(warning)
#define LOG_ERROR BOOST_LOG_TRIVIAL(error)
#define LOG_FATAL BOOST_LOG_TRIVIAL(fatal)

// Diagnostic logging: compile-time flag + runtime environment variable
// Compile with -DENABLE_DIAGNOSTICS to enable, or set SRDATALOG_DIAG_* env vars at runtime
#ifdef ENABLE_DIAGNOSTICS
#define LOG_DIAG BOOST_LOG_TRIVIAL(debug)
#define LOG_DIAG_ENABLED true
#else
// Runtime check: allow enabling via environment variable even without compile flag
// Note: This function is called at runtime, so expensive operations should be guarded
inline bool is_diagnostic_enabled(const char* env_var_name) {
  if (const char* env = std::getenv(env_var_name)) {
    return env[0] != '\0' && env[0] != '0';
  }
  return false;
}
// For runtime checks, use: if (is_diagnostic_enabled("SRDATALOG_DIAG_INTERSECT")) { ... }
#define LOG_DIAG BOOST_LOG_TRIVIAL(debug)
#define LOG_DIAG_ENABLED is_diagnostic_enabled("SRDATALOG_DIAG_INTERSECT")
#endif

}  // namespace SRDatalog::Logging

#else

// Logging disabled - all macros become no-ops
namespace SRDatalog::Logging {

inline void init_logging(bool to_console = true, bool to_file = false,
                         const std::string& log_file = "") {
  // No-op when logging is disabled
  (void)to_console;
  (void)to_file;
  (void)log_file;
}

}  // namespace SRDatalog::Logging

// Define macros as no-ops that compile to nothing
// Using a struct that discards output - compiler will optimize this away completely
struct NullStream {
  template <typename T>
  NullStream& operator<<(const T&) {
    return *this;
  }
};

#define LOG_TRACE NullStream()
#define LOG_DEBUG NullStream()
#define LOG_INFO NullStream()
#define LOG_WARNING NullStream()
#define LOG_ERROR NullStream()
#define LOG_FATAL NullStream()

// Diagnostic logging: disabled when logging is disabled
inline bool is_diagnostic_enabled(const char*) {
  return false;
}
#define LOG_DIAG_ENABLED false
#define LOG_DIAG NullStream()

#endif  // ENABLE_LOGGING
