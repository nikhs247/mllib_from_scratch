include(FetchContent)

# Download GoogleTest automatically
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

# Fetch it immediately
FetchContent_MakeAvailable(googletest)

#Create test executable
add_executable(unit_tests test_tensor.cpp)

# Link dependencies
# - gtest_main
# - mllib
target_link_libraries(unit_tests PRIVATE gtest_main mllib)

# Register with CTest
include(GoogleTest)
gtest_discover_tests(unit_tests)

