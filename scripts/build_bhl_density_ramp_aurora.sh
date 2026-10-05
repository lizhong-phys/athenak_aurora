#!/bin/bash
set -euo pipefail

source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-$source_dir/build_bhl_density_ramp}"
build_jobs="${BUILD_JOBS:-104}"

module load cmake
cmake -S "$source_dir" -B "$build_dir" \
  -D Athena_ENABLE_MPI=ON \
  -D CMAKE_CXX_COMPILER=icpx \
  -D Kokkos_ENABLE_SERIAL=ON \
  -D Kokkos_ARCH_INTEL_PVC=ON \
  -D Kokkos_ENABLE_OPENMP=ON \
  -D Kokkos_ENABLE_SYCL=ON \
  -D Kokkos_ENABLE_SYCL_RELOCATABLE_DEVICE_CODE=ON \
  -D PROBLEM=xin_bhl_density_ramp
cmake --build "$build_dir" --parallel "$build_jobs"
echo "Executable: $build_dir/src/athena"
