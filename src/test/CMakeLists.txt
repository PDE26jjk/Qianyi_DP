add_library(test)
file(GLOB BVH "bvh/*")
file(GLOB SDF "sdf/*")

target_sources(
   test PRIVATE 
   benchmarks.h
   sort_test.cu
   bvh_test.cu
   bvh_test2.cu
   color_graph_test.cu
   test.cpp
   test.h
   ${BVH}
   ${SDF}
)

find_package(Python REQUIRED COMPONENTS Development)
target_include_directories(test PRIVATE ${Python_INCLUDE_DIRS})
target_link_libraries(test PRIVATE pybind11::headers)

message(STATUS [[============================]])
message(STATUS "Test building with CUDA version ${CUDAToolkit_VERSION}")

target_include_directories(test PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_compile_options(test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
if(MSVC)
   set_target_properties(test PROPERTIES VS_GLOBAL_IncludePath "${PROJECT_SOURCE_DIR}/src")
endif()