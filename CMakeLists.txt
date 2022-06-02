cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(baseTest LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(FetchContent)
#cxxopts
FetchContent_Declare(
        cxxopts
        GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
        GIT_TAG c74846a891b3cc3bfa992d588b1295f528d43039 #v3.0.0
)
FetchContent_MakeAvailable(cxxopts)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${cxxopts_SOURCE_DIR}/include)
file(GLOB_RECURSE source_file LIST_DIRECTORIES false src/*.cpp src/*.c src/*.cu)
add_executable(${CMAKE_PROJECT_NAME} ${source_file})
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
find_package(Threads REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE cuda
        PRIVATE Threads::Threads)

#Handle cu file
set(NVCC_RELEASE_FLAGS "--resource-usage -Xptxas -dlcm=cg -lineinfo --generate-code='arch=compute_86,code=[compute_86,sm_86]'")
set(NVCC_DEBUG_FLAGS "--device-debug --resource-usage --ptxas-options='-O0' -Xptxas -dlcm=cg -lineinfo")
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${NVCC_DEBUG_FLAGS}")
else()
    string(TOLOWER ${CMAKE_BUILD_TYPE} build_type)
    if (${build_type} STREQUAL "release")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${NVCC_RELEASE_FLAGS}")
    else()
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${NVCC_DEBUG_FLAGS}")
    endif()
endif()

message(STATUS "CMAKE_CUDA_FLAGS:${CMAKE_CUDA_FLAGS}")
if(APPLE)
    message(FATAL_ERROR "Unsupported on mac")
endif()
