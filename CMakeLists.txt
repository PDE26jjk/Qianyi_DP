cmake_minimum_required(VERSION 3.24)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_BUILD_TYPE Debug)

if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi")
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} /DEBUG")
endif()

project(Qianyi_DP LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

if(NOT Python_EXECUTABLE)
    find_package(Python REQUIRED COMPONENTS Development)
endif()

# include_directories("${Python_INCLUDE_DIRS}")
# include_directories("extern/pybind11/include")
message(STATUS "CUDA_VERSION: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDA_TOOLKIT_ROOT_DIR: ${CUDAToolkit_ROOT_DIR}")

message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3_LIBRARIES: ${Python3_LIBRARIES}")

add_subdirectory(extern/pybind11)

# set(test_project "${PROJECT_NAME}_test")
# add_executable(${test_project} test.cpp)
set_source_files_properties(*.cu PROPERTIES LANGUAGE CUDA)

PYBIND11_ADD_MODULE(${PROJECT_NAME} src/main.cu src/main.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE
    ${Python_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11/include
)
target_sources(${PROJECT_NAME} PRIVATE
    src/simulator_interface.h
    src/simulator_interface.cpp
 )

add_subdirectory(src/sample)
add_subdirectory(src/gDel2D)
add_subdirectory(src/simulation)

target_include_directories(${PROJECT_NAME} PRIVATE src)
target_link_libraries(${PROJECT_NAME}
    PRIVATE sample
    PRIVATE gDel2D
    PRIVATE simulation
)

if(MSVC)
    add_library(common INTERFACE)
    file(GLOB COMMON_HEADERS "src/common/*.h")
    file(GLOB COMMON_HEADERS2 "src/common/*.cuh")
    target_sources(common INTERFACE ${COMMON_HEADERS} ${COMMON_HEADERS2})
    target_link_libraries(${PROJECT_NAME}
        PRIVATE common
    )
endif()



