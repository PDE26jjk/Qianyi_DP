add_library(simulation)
file(GLOB DYNAMIC_SOLVER "dynamics/*.cuh")
file(GLOB CONTACT "contact/*.cuh")
file(GLOB CUDA_TOOLS "cuda_tools/*")
file(GLOB SDF "sdf/*")
file(GLOB COLOR_GRAPH "color_graph/*")
target_sources(
    simulation PRIVATE
    simulator.h
    simulator.cu
    sewing.cu
    picker.cu
    pin.cu
    ${DYNAMIC_SOLVER}
    ${CONTACT}
    ${CUDA_TOOLS}
    ${SDF}
    ${COLOR_GRAPH}
    contact/lbvh.cu
    # collision_Wu2021.cu
    collision.cu
    geometry.cuh
    geometry.cu
    subspace.cu
    solver_base.cuh
    solver_base.cu
    solver_explicit.cuh
    solver_explicit.cu
    solver_XPBD.cuh
    solver_XPBD.cu
    solver_PDNewton.cuh
    solver_PDNewton.cu
    solver_VBD.cuh
    solver_VBD.cu
    linear/solver_linear.cuh
    linear/solver_linear.cu
    linear/solver_PCG.cu
    # solver_PNCG.cuh
    # solver_PNCG.cu
    # solver_Chebyshev.cuh
    # solver_Chebyshev.cu
    constraint.cuh
    geometric_operator.cuh
)
target_include_directories(simulation PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_compile_options(simulation PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
if(MSVC)
   set_target_properties(simulation PROPERTIES VS_GLOBAL_IncludePath "${PROJECT_SOURCE_DIR}/src")
endif()