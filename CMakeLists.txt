cmake_minimum_required(VERSION 3.24)
project(Qianyi_DP LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# set(CMAKE_CUDA_ARCHITECTURES native)
# set(CUDA_ARCHS 
#     sm_75  # Turing (GTX 16xx, RTX 20xx)
#     sm_80  # Ampere (A100)
#     sm_86  # Ampere (RTX 30xx)
#     sm_89  # Ada Lovelace (RTX 40xx)
#     sm_90  # Hopper (H100)
# )
# set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_BUILD_TYPE Debug)

if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi")
    # set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G") # it make dll very big
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} /DEBUG")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/MP>)
endif()


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
    src/geometry_interface.h
    src/geometry_interface.cpp
    src/pattern_helper_interface.h
    src/pattern_helper_interface.cpp
    src/simulator_interface.h
    src/simulator_interface.cpp
 )

add_subdirectory(src/geometry)
add_subdirectory(src/gDel2D)
add_subdirectory(src/simulation)
add_subdirectory(src/graphics)

target_include_directories(${PROJECT_NAME} PRIVATE src)
target_link_libraries(${PROJECT_NAME}
    PRIVATE geometry
    PRIVATE gDel2D
    PRIVATE simulation
    PRIVATE graphics
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



