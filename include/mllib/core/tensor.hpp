#ifndef MLLIB_CORE_TENSOR_HPP
#define MLLIB_CORE_TENSOR_HPP

#include <vector>
#include <numeric>
#include <iostream>
#include <type_traits>
#include <stdexcept>

namespace mllib {
    namespace core {

        template<typename T>
        class Tensor {
            static_assert(std::is_arithmetic_v<T>, "Tensor only supports arithmetic types.");
            
            private:
                std::vector<T> m_data;          // Flat data storage
                std::vector<size_t> m_shape;    // Shape of the tensor

            public:

                // ---CONSTRUCTORS---

                // Default constructor
                Tensor() = default;

                // Constructor with shape: Tensor({2, 3, 4}) creates a 2x3x4 tensor
                Tensor(const std::vector<size_t>& shape) : m_shape(shape) {
                    size_t total_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
                    m_data.resize(total_size);
                }

                // Constructor with shape and initial value: Tensor({2, 3}, 0.0) creates a 2x3 tensor filled with 0.0
                Tensor(const std::vector<size_t>& shape, T initial_value) : m_shape(shape) {
                    size_t total_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
                    m_data.resize(total_size, initial_value);
                }

                // -- DATA ACCESS---

                // Get total elements
                [[nodiscard]] size_t size() const {
                    return m_data.size();
                }

                // Get shape
                [[nodiscard]] const std::vector<size_t>& shape() const {
                    return m_shape;
                }

                // Access raw data
                T* data() {
                    return m_data.data();
                }
                const T* data() const {
                    return m_data.data();
                }

                // --- INDEXING ---

                // 1D Access
                T& operator[](size_t index) {
                    return m_data[index];
                }
                const T& operator[](size_t index) const {
                    return m_data[index];
                }

                // ND Access
                T& at(const std::vector<size_t>& indices){
                    size_t flat_index = calculate_flat_index(indices);
                    return m_data[flat_index];
                }

                const T& at(const std::vector<size_t>& indices) const{
                    size_t flat_index = calculate_flat_index(indices);
                    return m_data[flat_index];
                }

                // Element wise addition
                Tensor<T> operator+(const Tensor<T>& other) const {
                    if(m_shape != other.m_shape){
                        throw std::invalid_argument("Tensor shapes do not match for addition.");
                    }

                    Tensor<T> result(m_shape);

                    for(size_t i = 0; i < m_data.size(); i++) {
                        result.m_data[i] = m_data[i] + other.m_data[i];
                    }

                    return result;
                }

                // Matrix multiplication
                Tensor<T> matmul(const Tensor<T>& other) const {
                    if(m_shape.size() != 2 || other.m_shape.size() != 2) {
                        throw std::invalid_argument("Both tensors must be 2D for matrix multiplication.");
                    }
                    if(m_shape[1] != other.m_shape[0]) {
                        throw std::invalid_argument("Inner dimensions do not match for matrix multiplication.");
                    }

                    // rows x inner_dim, inner_dim x cols => rows x cols
                    size_t rows = m_shape[0];
                    size_t cols = other.m_shape[1];
                    size_t inner_dim = m_shape[1];

                    // Intialize result tensor
                    // TODO: Optimize with better memory management
                    Tensor<T> result({rows, cols}, static_cast<T>(0));
                    for(size_t i = 0; i < rows; i++) {
                        for(size_t k = 0; k < inner_dim; k++) {
                            T val_A = this->at({i, k});
                            for(size_t j = 0; j < cols; j++) {
                                T val_B = other.at({k, j});
                                result.at({i, j}) += val_A * val_B;
                            }
                        }
                    }

                    return result;

                }

            private:
                size_t calculate_flat_index(const std::vector<size_t>& indices) const {
                    if(indices.size() != m_shape.size()) {
                        throw std::out_of_range("Number of indices does not match tensor dimesnions.");
                    }

                    size_t flat_index = 0;
                    size_t stride = 1;

                    // Iterate backwards to calculate flat index
                    // Row major layout
                    for(int i = m_shape.size() - 1; i >= 0; i--) {
                        flat_index += indices[i] * stride;
                        stride *= m_shape[i];
                    }
                    return flat_index;
                }

            };
    } // namespace core
} // namespace mllib

#endif // MLLIB_CORE_TENSOR_HPP