cmake_minimum_required(VERSION 3.20)
project(tegula LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD 17)

find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_C NB_STATIC
  src/tegula/_C/bindings.mm
  src/tegula/_C/tegula_rt.mm)
target_compile_options(_C PRIVATE -fobjc-arc)
target_link_libraries(_C PRIVATE "-framework Metal" "-framework Foundation")

install(TARGETS _C LIBRARY DESTINATION tegula)
