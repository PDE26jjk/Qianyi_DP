add_library(sample)
target_sources(
   sample PRIVATE
   sample_points.cu
)
target_include_directories(sample PUBLIC ${PROJECT_SOURCE_DIR}/src)
if(MSVC)
    # A hack to set include path to VS, taught by deepseek. 
   set_target_properties(sample PROPERTIES VS_GLOBAL_IncludePath "${PROJECT_SOURCE_DIR}/src")
endif()
# get_target_property(VC_IncludePath sample VS_GLOBAL_IncludePath)
# message(STATUS "------------------------------${PROJECT_SOURCE_DIR}/src")
# message(STATUS "------------------------------VC_IncludePath ${VC_IncludePath}")
