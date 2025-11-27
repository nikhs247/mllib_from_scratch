cmake_minimum_required(VERSION 3.14)
project(mllib_from_scratch VERSION 1.0.0 LANGUAGES CXX)

# Standard Settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Main library target
add_library(mllib INTERFACE)
target_include_directories(mllib INTERFACE include)

# Setup testing
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()