#pragma once

// Separate header for Boost.Log includes to ensure they're always at global scope
// This prevents Boost's internal code from looking in SRDatalog::boost namespace
// when this header is included from within a namespace context.

#ifdef ENABLE_LOGGING
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/utility/setup/file.hpp>
#endif
