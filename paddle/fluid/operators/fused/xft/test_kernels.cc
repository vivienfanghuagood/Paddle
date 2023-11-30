#include <stdio.h>
#include "layers/matmul_helper.h"
#include "layers/layer_norm.h"
#include "./common/float16.h"

template <typename T>
void layer_norm_ref(T *output, const T *input, const T *gamma, const T *beta, int rows, int cols,
        int iStride = -1, int oStride = -1, const float epsilon = 1e-5) {
    if (iStride == -1) iStride = cols;
    if (oStride == -1) oStride = cols;

    // Iterate over rows
    for (int i = 0; i < rows; ++i) {
        // Compute mean
        float mean = 0.0;
        for (int j = 0; j < cols; ++j) {
            mean += input[i * iStride + j];
        }
        mean /= cols;

        // Compute variance
        float variance = 0.0;
        for (int j = 0; j < cols; ++j) {
            T diff = input[i * iStride + j] - mean;
            variance += diff * diff;
        }
        variance /= cols;

        // Normalize
        T inv_std_dev = static_cast<T>(1.0 / std::sqrt(variance + epsilon));
        for (int j = 0; j < cols; ++j) {
            output[i * oStride + j] = gamma[j] * (input[i * iStride + j] - mean) * inv_std_dev + beta[j];
        }
    }
}


template <typename T>
void compareLayerNorm(int rows, int cols) {

    T *input = (T *)aligned_alloc(64, rows * cols * sizeof(T));
    T *gamma = (T *)aligned_alloc(64, cols * sizeof(T));
    T *beta = (T *)aligned_alloc(64, cols * sizeof(T));
    T *ourOutput = (T *)aligned_alloc(64, rows * cols * sizeof(T));
    T *refOutput = (T *)aligned_alloc(64, rows * cols * sizeof(T));

    for (int i = 0; i < rows * cols; ++i) {
        input[i] = static_cast<T>(1.0f * rand() / RAND_MAX);
    }

    for (int i = 0; i < cols; ++i) {
        gamma[i] = static_cast<T>(1.0f);
    }

    for (int i = 0; i < cols; ++i) {
        beta[i] = static_cast<T>(0.0f);
    }

    if constexpr (std::is_same<T, float>::value) {
        xft::invokeLayerNorm(xft::DataType::fp32, (void *)ourOutput, (const void *)input, (const void *)gamma,
                (const void *)beta, rows, cols);
        layer_norm_ref<float>(refOutput, (const T *)input, (const T *)gamma, (const T *)beta, rows, cols);
    } else if constexpr (std::is_same<T, float16_t>::value) {
        xft::invokeLayerNorm(xft::DataType::fp16, (void *)ourOutput, (const void *)input, (const void *)gamma,
                (const void *)beta, rows, cols);
        layer_norm_ref<float16_t>(refOutput, (const T *)input, (const T *)gamma, (const T *)beta, rows, cols);
    } else if constexpr (std::is_same<T, bfloat16_t>::value) {
        xft::invokeLayerNorm(xft::DataType::bf16, (void *)ourOutput, (const void *)input, (const void *)gamma,
                (const void *)beta, rows, cols);
        layer_norm_ref<bfloat16_t>(refOutput, (const T *)input, (const T *)gamma, (const T *)beta, rows, cols);
    }

    for (int i = 0; i < rows * cols; ++i) {
        printf("ourOutput[%d] = %f\t refOutput[%d] = %f\n", i, ourOutput[i], i, refOutput[i]);
        // EXPECT_LT(((float)refOutput[i] - (float)ourOutput[i]), 0.01);
    }

    free(input);
    free(gamma);
    free(beta);
    free(ourOutput);
    free(refOutput);
}

void compareGEMM(int M, int N, int K) {
    const float A[M * K] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float B[K * N + 100] = {0.5f, 0.5f, 1.0f, 1.0f, 1.5f, 1.5f, 2.0f, 2.0f};
    float16_t B_float16[K * N + 100] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float B_float32[K * N + 100] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const float scale_B[N] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float zero_B[N] = {0.0f, 0.0f, 0.0f, 0.0f};
    float C[M * N] = {0.0f};

    const float expected_C[M * N] = {3.5f, 3.5f, 5.0f, 5.0f, 7.5f, 7.5f, 11.0f, 11.0f, 11.5f, 11.5f, 17.0f, 17.0f};
    
    float16_t::cvt_float_to_float16(B, B_float16, K * N);
    float16_t::cvt_float16_to_float(B_float16, B_float32, K * N);
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("B_converted[%d * %d] = %f, origin = %f\n", i, j, B[i * N + j], B_float32[i * N + j]);
        }
    }

    // small_sgemm(M, N, K, A, K, B, N, C, N);
    MMHelper::compute<float>(false, M, N, K, 1.0f, A, K, B_float32, scale_B, zero_B, 0.0, C, N);
    // MMHelper::compute<float16_t>(false, M, N, K, 1.0f, A, K, B_float16, scale_B, zero_B, 0.0, C, N);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("C[%d * %d] = %f\t %f\n", i, j, C[i * N + j], expected_C[i * N + j]);
            // EXPECT_FLOAT_EQ(C[i * N + j], expected_C[i * N + j]);
        }
    }
}
int main(){
    // const int M = 3;
    // const int N = 4;
    // const int K = 2;

    compareGEMM(3, 4, 2);
    compareLayerNorm<float>(10, 8);
}