add_library(graphics)

target_sources(
   graphics PRIVATE 
   graphics.h
   # t1.cpp
   # t2.cpp
   t3.cpp
)
if (NOT IS_DIRECTORY $ENV{VULKAN_SDK})
    message(WARNING "Vulkan SDK not detected, if installed please set VULKAN_SDK env var or manually provide libraries")
else()
    message(STATUS "Found Vulkan SDK in $ENV{VULKAN_SDK}")
endif()
include_directories(SYSTEM $ENV{VULKAN_SDK}/include)

find_package(Vulkan REQUIRED)

if(MSVC)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif()

target_link_libraries(graphics PUBLIC Vulkan::Vulkan)