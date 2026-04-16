cmake_minimum_required(VERSION 3.20)

project(Boltzmann1DBGKSolver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BOLTZMANN_ENABLE_OPENMP "Enable OpenMP parallelism when available" ON)

add_executable(Boltzmann1DBGKSolver
    Boltzmann_1D_Solver/include/json/jsoncpp.cpp
    Boltzmann_1D_Solver/solver/main.cpp
    Boltzmann_1D_Solver/utils/Norma.cpp
)

target_include_directories(Boltzmann1DBGKSolver PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Boltzmann_1D_Solver/include
    ${CMAKE_CURRENT_SOURCE_DIR}/Boltzmann_1D_Solver/interface
    ${CMAKE_CURRENT_SOURCE_DIR}/Boltzmann_1D_Solver/solver
    ${CMAKE_CURRENT_SOURCE_DIR}/Boltzmann_1D_Solver/utils
)

if(BOLTZMANN_ENABLE_OPENMP)
    find_package(OpenMP COMPONENTS CXX QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(Boltzmann1DBGKSolver PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(Boltzmann1DBGKSolver PRIVATE BOLTZMANN_HAS_OPENMP=1)
    else()
        message(STATUS "OpenMP not found. Building without OpenMP support.")
    endif()
endif()

if(MSVC)
    target_compile_options(Boltzmann1DBGKSolver PRIVATE /W4)
else()
    target_compile_options(Boltzmann1DBGKSolver PRIVATE -Wall -Wextra -Wpedantic)
endif()
