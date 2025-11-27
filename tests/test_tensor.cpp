#include <gtest/gtest.h>
#include "mllib/core/tensor.hpp"

// Test Case 1: Check Shap Initialization
TEST(TensorTest, Initialization) {
    mllib::core::Tensor<float> t({2, 3}, 5.0f);

    // Check shape
    EXPECT_EQ(t.size(), 6);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
}

// Test Case 2: Checking Data Access
TEST(TensorTest, ReadWrite) {
    mllib::core::Tensor<float> t({2, 2}, 0.0f);

    // Write data
    t.at({0, 1}) = 42.5f;

    // Read data
    EXPECT_FLOAT_EQ(t.at({0, 1}), 42.5f);
    EXPECT_FLOAT_EQ(t.at({1, 1}), 0.0f); // Default value
}

// Test Case 3: Checking exceptions safety
TEST(TensorTest, OutOfBounds) {
    mllib::core::Tensor<float> t({2, 2});

    // Throws exception as 3 dimesnions are provided for a 2D tensor
    EXPECT_THROW(t.at({0, 1, 2}), std::out_of_range);
}