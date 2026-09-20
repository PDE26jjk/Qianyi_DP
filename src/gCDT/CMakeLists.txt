# gCDT: GPU constrained Delaunay triangulation (see README.md in this folder).
#
# Vendored from https://github.com/yingtix/gCDT at c5252b1 plus the local
# robustness work; the demo entry point and the scene reader were removed, the
# library entry point is gcdtBuildMesh() in pipeline.h.

add_library(gCDT STATIC
    addConstraints.cu
    boostmap.cu
    fastDelaunay.cu
    kernel.cu
    rawMesh.cu
    utils.cu
    validate.cu
    pipeline.cu
    MemCache.cpp

    include/addConstraints.h
    include/boostmap.h
    include/fastDelaunay.h
    include/internal_stages.h
    include/MemCache.h
    include/math_base.cuh
    include/pipeline.h
    include/rawMesh.h
    include/utils.cuh
    include/validate.h
)

target_include_directories(gCDT PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(gCDT PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(gCDT PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
)

# Static CUDA runtime: the module must stay a single .pyd without a runtime DLL
# dependency, exactly like the rest of the project.
target_link_libraries(gCDT PUBLIC CUDA::cudart_static)
