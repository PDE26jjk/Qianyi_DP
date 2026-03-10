add_library(simulation)
file(GLOB DYNAMIC_SOLVER "dynamics/*.cuh")
target_sources(
   simulation PRIVATE
   simulator.h
   simulator.cu
   collision.cu
   picker.cu
   ${DYNAMIC_SOLVER}
   solver_base.cuh
   solver_base.cu
   solver_PCG.cuh
   solver_PCG.cu
   constraint.cuh
   geometric_operator.cuh
   sewing.cu
)
target_include_directories(simulation PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_compile_options(simulation PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
if(MSVC)
   set_target_properties(simulation PROPERTIES VS_GLOBAL_IncludePath "${PROJECT_SOURCE_DIR}/src")
endif()