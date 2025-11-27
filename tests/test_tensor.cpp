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

// Test Case 4: Tensor Addition
TEST(TensorTest, Addition) {
    mllib::core::Tensor<int> a({2, 2}, 1);
    mllib::core::Tensor<int> b({2, 2}, 2);
    mllib::core::Tensor<int> c = a + b;
    EXPECT_EQ(c.at({0, 0}), 3);
    EXPECT_EQ(c.at({1, 1}), 3);
}

// Test Case 5: Matrix Multiplication
TEST(TensorTest, MatMul) {
    // Matrix A (2x2)
    mllib::core::Tensor<float> A({2, 2});
    A.at({0, 0}) = 1; A.at({0, 1}) = 2;
    A.at({1, 0}) = 3; A.at({1, 1}) = 4;

    // Matrix B (2x2)
    mllib::core::Tensor<float> B({2, 2});
    B.at({0, 0}) = 2; B.at({0, 1}) = 0;
    B.at({1, 0}) = 1; B.at({1, 1}) = 2;

    // Resultant Matrix C = A * B
    mllib::core::Tensor<float> C = A.matmul(B);

    EXPECT_EQ(C.shape()[0], 2);
    EXPECT_EQ(C.shape()[1], 2);

    EXPECT_FLOAT_EQ(C.at({0, 0}), 4);  // 1*2 + 2*1
    EXPECT_FLOAT_EQ(C.at({0, 1}), 4);  // 1*0 + 2*2
    EXPECT_FLOAT_EQ(C.at({1, 0}), 10); // 3*2 + 4*1
    EXPECT_FLOAT_EQ(C.at({1, 1}), 8);  // 3*0 + 4*2
}