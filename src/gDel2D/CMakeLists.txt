# project(gDel2d)
# find_package(CUDAToolkit REQUIRED)
# Visualization dependencies - disabled
# find_package(OpenGL)
# find_package(GLEW REQUIRED)
# find_package(GLUT REQUIRED)
# set(CMAKE_BUILD_TYPE Release)
# set(CUDA_NVCC_FLAGS
#     ${CUDA_NVCC_FLAGS};
#     -gencode arch=compute_86,code=sm_86
#     -gencode arch=compute_50,code=sm_50)
#add_definitions(-DREAL_TYPE_FP32)
# include_directories(
#     src
#     )
add_library(gDel2D)
target_sources(
    gDel2D PRIVATE
      DelaunayChecker.cpp
      HashTable.cpp
      # RandGen.cpp
      # Visualizer.cpp    # Disabled - requires OpenGL/GLEW/GLUT
      InputCreator.cpp
      gDel2D/CPU/PredWrapper.cpp
      gDel2D/CPU/predicates.cpp
      Main.cpp
      gDel2D/GPU/GpuDelaunay.cu
      gDel2D/GPU/ThrustWrapper.cu
      gDel2D/GPU/KerPredicates.cu
      gDel2D/GPU/KerDivision.cu
      gDel2D/GPU/SmallCounters.cu
   )
# set(CMAKE_VERBOSE_MAKEFILE "ON")
# target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES})  # Disabled visualization

