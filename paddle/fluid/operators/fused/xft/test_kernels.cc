#include <stdio.h>
#include "matmul_helper.h"

template<typename T>
void init_value(int number, T data, T* ptr){
    for(int i = 0; i < number; ++i){
        ptr[i] = data;
    }
}

template<typename T>
void print_value(int number, T* ptr){
    for(int i = 0; i < number; ++i){
        std::cout << ptr[i] << " ";
    }
}

int main(){
    const int M = 3;
    const int N = 4;
    const int K = 2;

    const float A[M * K] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float B[K * N] = {0.5f, 0.5f, 1.0f, 1.0f, 1.5f, 1.5f, 2.0f, 2.0f};
    const float scale_B[N] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float zero_B[N] = {0.0f, 0.0f, 0.0f, 0.0f};
    float C[M * N] = {0.0f};

    const float expected_C[M * N] = {3.5f, 3.5f, 5.0f, 5.0f, 7.5f, 7.5f, 11.0f, 11.0f, 11.5f, 11.5f, 17.0f, 17.0f};

    // small_sgemm(M, N, K, A, K, B, N, C, N);
    MMHelper::compute<float>(false, M, N, K, 1.0f, A, K, B, scale_B, zero_B, 0.0, C, N);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("C[%d * %d] = %f\t %f", i, j, C[i * N + j], expected_C[i * N + j]);
            // EXPECT_FLOAT_EQ(C[i * N + j], expected_C[i * N + j]);
        }
    }
}