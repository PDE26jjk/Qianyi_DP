add_library(geometry)
target_sources(
   geometry PRIVATE
   sample_points.cu
   sampler.h
   lbvh_2d.h
   lbvh_2d.cu
   mapbox/earcut.hpp
   triangulator.h
   triangulator.cpp
   pattern_helper.h
   pattern_helper.cu
)
target_include_directories(geometry PUBLIC ${PROJECT_SOURCE_DIR}/src)
if(MSVC)
    # A hack to set include path to VS, taught by deepseek. 
   set_target_properties(geometry PROPERTIES VS_GLOBAL_IncludePath "${PROJECT_SOURCE_DIR}/src")
endif()
# get_target_property(VC_IncludePath geometry VS_GLOBAL_IncludePath)
# message(STATUS "------------------------------${PROJECT_SOURCE_DIR}/src")
# message(STATUS "------------------------------VC_IncludePath ${VC_IncludePath}")
