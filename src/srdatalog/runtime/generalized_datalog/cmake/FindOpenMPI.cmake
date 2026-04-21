# Supplementary OpenMPI Detection Module
# This module enhances CMake's built-in MPI detection with OpenMPI-specific features
# Used as a fallback if the standard find_package(MPI) doesn't find OpenMPI

include(FindPackageHandleStandardArgs)

# Try to find OpenMPI using pkg-config first
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
  pkg_check_modules(PC_MPI ompi-cxx QUIET)
endif()

# Look for mpicc/mpicxx
find_program(MPI_C_COMPILER
  NAMES mpicc
  HINTS ${PC_MPI_BINDIR} /usr/bin /usr/local/bin
)

find_program(MPI_CXX_COMPILER
  NAMES mpicxx mpiCC mpic++
  HINTS ${PC_MPI_BINDIR} /usr/bin /usr/local/bin
)

# Find MPI headers
find_path(MPI_C_INCLUDE_DIR
  NAMES mpi.h
  HINTS ${PC_MPI_INCLUDEDIR} /usr/include /usr/local/include
)

find_path(MPI_CXX_INCLUDE_DIR
  NAMES mpi.h
  HINTS ${PC_MPI_INCLUDEDIR} /usr/include /usr/local/include
)

# Find MPI libraries
find_library(MPI_C_LIB
  NAMES mpi
  HINTS ${PC_MPI_LIBDIR} /usr/lib /usr/local/lib
)

find_library(MPI_CXX_LIB
  NAMES mpi_cxx
  HINTS ${PC_MPI_LIBDIR} /usr/lib /usr/local/lib
)

# Handle the standard arguments
find_package_handle_standard_args(OpenMPI
  FOUND_VAR OpenMPI_FOUND
  REQUIRED_VARS MPI_CXX_COMPILER MPI_C_INCLUDE_DIR MPI_CXX_INCLUDE_DIR
  VERSION_VAR OpenMPI_VERSION
)

if(OpenMPI_FOUND)
  # Create interface targets for better CMake integration
  if(NOT TARGET MPI::MPI_C)
    add_library(MPI::MPI_C INTERFACE IMPORTED)
    set_target_properties(MPI::MPI_C PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${MPI_C_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES "${MPI_C_LIB}"
    )
  endif()

  if(NOT TARGET MPI::MPI_CXX)
    add_library(MPI::MPI_CXX INTERFACE IMPORTED)
    set_target_properties(MPI::MPI_CXX PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES "${MPI_C_LIB};${MPI_CXX_LIB}"
    )
  endif()

  message(STATUS "OpenMPI found:")
  message(STATUS "  C Compiler: ${MPI_C_COMPILER}")
  message(STATUS "  CXX Compiler: ${MPI_CXX_COMPILER}")
  message(STATUS "  Include Dir: ${MPI_C_INCLUDE_DIR}")
endif()

mark_as_advanced(
  MPI_C_COMPILER
  MPI_CXX_COMPILER
  MPI_C_INCLUDE_DIR
  MPI_CXX_INCLUDE_DIR
  MPI_C_LIB
  MPI_CXX_LIB
)

