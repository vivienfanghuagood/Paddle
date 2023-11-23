// Copyright (c) 2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
#pragma once
#include <map>
#include <tuple>

#include "third_party/ig/common/float16.h"
#include "third_party/ig/common/bfloat16.h"
// #include "third_party/ig/ig_bgemm_f32bf16f32.h"
#include "third_party/ig/ig_hgemm_f32f16f32.h"
#include "third_party/ig/ig_hgemm_f16f16f32.h"
#include "third_party/ig/ig_hgemm_f32i8f32.h"
#include "third_party/ig/ig_sgemm.h"
#include "third_party/ig/ig_sgemm_f32f16f32.h"
#include "third_party/ig/ig_sgemm_f32i8f32.h"
#include "./common/my_types.h"
#include "./util/split_util.h"

#define AVX512_FP32_WEIGHT_ONLY_FP16 1
#define AVX512_FP32_WEIGHT_ONLY_INT8 1

class MMHelper {
public:
 
    template <typename WeiT>
    static void compute(bool transA, int M, int N, int K, float alpha, const float *A, int lda, const WeiT *packedB,
            const float *scaleB, const float *zeroB, float beta, float *C, int ldc) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            ig_sgemm_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            // TimeLine t("ig_sgemm_f32f16f32_compute");
            ig_sgemm_f32f16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            // TimeLine t("ig_hgemm_f32f16f32_compute");
            ig_hgemm_f32f16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            // TimeLine t("ig_sgemm_f32bf16f32_compute");
            ig_sgemm_f32bf16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                // TimeLine t("ig_amx_sgemm_f32bf16f32_compute");
                ig_amx_sgemm_f32bf16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
            } else {
                // TimeLine t("ig_bgemm_f32bf16f32_compute");
                ig_bgemm_f32bf16f32_compute(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            // TimeLine t("ig_sgemm_f32i8f32_compute");
            ig_sgemm_f32i8f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            // TimeLine t("ig_hgemm_f32i8f32_compute");
            ig_hgemm_f32i8f32_compute(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename WeiT>
    static void compute_bias(bool transA, int M, int N, int K, float alpha, const float *A, int lda,
            const WeiT *packedB, const float *scaleB, const float *zeroB, float beta, float *C, int ldc,
            const float *bias) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            // TimeLine t("ig_sgemm_compute_biasadd");
            ig_sgemm_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            // TimeLine t("ig_sgemm_f32f16f32_compute_biasadd");
            ig_sgemm_f32f16f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            // TimeLine t("ig_hgemm_f32f16f32_compute_biasadd");
            ig_hgemm_f32f16f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            // TimeLine t("ig_sgemm_f32bf16f32_compute_biasadd");
            ig_sgemm_f32bf16f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                // TimeLine t("ig_amx_sgemm_f32bf16f32_compute_biasadd");
                ig_amx_sgemm_f32bf16f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
            } else {
                // TimeLine t("ig_bgemm_f32bf16f32_compute_biasadd");
                ig_bgemm_f32bf16f32_compute_biasadd(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            // TimeLine t("ig_sgemm_f32i8f32_compute_biasadd");
            ig_sgemm_f32i8f32_compute_biasadd(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            // TimeLine t("ig_hgemm_f32i8f32_compute_biasadd");
            ig_hgemm_f32i8f32_compute_biasadd(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename WeiT>
    static void compute_biasadd_relu(bool transA, int M, int N, int K, float alpha, const float *A, int lda,
            const WeiT *packedB, const float *scaleB, const float *zeroB, float beta, float *C, int ldc,
            const float *bias) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            // TimeLine t("ig_sgemm_compute_biasadd_relu");
            ig_sgemm_compute_biasadd_relu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            // TimeLine t("ig_sgemm_f32f16f32_compute_biasadd_relu");
            ig_sgemm_f32f16f32_compute_biasadd_relu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            // TimeLine t("ig_hgemm_f32f16f32_compute_biasadd_relu");
            ig_hgemm_f32f16f32_compute_biasadd_relu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            // TimeLine t("ig_sgemm_f32bf16f32_compute_biasadd_relu");
            ig_sgemm_f32bf16f32_compute_biasadd_relu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                // TimeLine t("ig_amx_sgemm_f32bf16f32_compute_biasadd_relu");
                ig_amx_sgemm_f32bf16f32_compute_biasadd_relu(
                        transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
            } else {
                // TimeLine t("ig_bgemm_f32bf16f32_compute_biasadd_relu");
                ig_bgemm_f32bf16f32_compute_biasadd_relu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias);
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            // TimeLine t("ig_sgemm_f32i8f32_compute_biasadd_relu");
            ig_sgemm_f32i8f32_compute_biasadd_relu(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            // TimeLine t("ig_hgemm_f32i8f32_compute_biasadd_relu");
            ig_hgemm_f32i8f32_compute_biasadd_relu(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename WeiT>
    static void compute_silu(bool transA, int M, int N, int K, float alpha, const float *A, int lda,
            const WeiT *packedB, const float *scaleB, const float *zeroB, float beta, float *C, int ldc) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            // TimeLine t("ig_sgemm_compute_silu");
            ig_sgemm_compute_silu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            // TimeLine t("ig_sgemm_f32f16f32_compute_silu");
            ig_sgemm_f32f16f32_compute_silu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            // TimeLine t("ig_hgemm_f32f16f32_compute_silu");
            ig_hgemm_f32f16f32_compute_silu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            // TimeLine t("ig_sgemm_f32bf16f32_compute_silu");
            ig_sgemm_f32bf16f32_compute_silu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                // TimeLine t("ig_amx_sgemm_f32bf16f32_compute_silu");
                ig_amx_sgemm_f32bf16f32_compute_silu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
            } else {
                // TimeLine t("ig_bgemm_f32bf16f32_compute_silu");
                ig_bgemm_f32bf16f32_compute_silu(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc);
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            // TimeLine t("ig_sgemm_f32i8f32_compute_silu");
            ig_sgemm_f32i8f32_compute_silu(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            // TimeLine t("ig_hgemm_f32i8f32_compute_silu");
            ig_hgemm_f32i8f32_compute_silu(transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename WeiT>
    static void compute_resmul(bool transA, int M, int N, int K, float alpha, const float *A, int lda,
            const WeiT *packedB, const float *scaleB, const float *zeroB, float beta, float *C, int ldc,
            const float *res, int ldres) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            // TimeLine t("ig_sgemm_compute_resmul");
            ig_sgemm_compute_resmul(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, res, ldres);
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            // TimeLine t("ig_sgemm_f32f16f32_compute_resmul");
            ig_sgemm_f32f16f32_compute_resmul(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, res, ldres);
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            // TimeLine t("ig_hgemm_f32f16f32_compute_resmul");
            ig_hgemm_f32f16f32_compute_resmul(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, res, ldres);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            // TimeLine t("ig_sgemm_f32bf16f32_compute_resmul");
            ig_sgemm_f32bf16f32_compute_resmul(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, res, ldres);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                // TimeLine t("ig_amx_sgemm_f32bf16f32_compute_resmul");
                ig_amx_sgemm_f32bf16f32_compute_resmul(
                        transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, res, ldres);
            } else {
                // TimeLine t("ig_bgemm_f32bf16f32_compute_resmul");
                ig_bgemm_f32bf16f32_compute_resmul(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, res, ldres);
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            // TimeLine t("ig_sgemm_f32i8f32_compute_resmul");
            ig_sgemm_f32i8f32_compute_resmul(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, res, ldres);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            // TimeLine t("ig_hgemm_f32i8f32_compute_resmul");
            ig_hgemm_f32i8f32_compute_resmul(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, res, ldres);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename WeiT>
    static void compute_residential(bool transA, int M, int N, int K, float alpha, const float *A, int lda,
            const WeiT *packedB, const float *scaleB, const float *zeroB, float beta, float *C, int ldc,
            const float *bias, const float *res, int ldres) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            // TimeLine t("ig_sgemm_compute_residential");
            ig_sgemm_compute_residential(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres);
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            // TimeLine t("ig_sgemm_f32f16f32_compute_residential");
            ig_sgemm_f32f16f32_compute_residential(
                    transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres);
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            // TimeLine t("ig_hgemm_f32f16f32_compute_residential");
            ig_hgemm_f32f16f32_compute_residential(
                    transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            // TimeLine t("ig_sgemm_f32bf16f32_compute_residential");
            ig_sgemm_f32bf16f32_compute_residential(
                    transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                // TimeLine t("ig_amx_sgemm_f32bf16f32_compute_residential");
                ig_amx_sgemm_f32bf16f32_compute_residential(
                        transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres);
            } else {
                // TimeLine t("ig_bgemm_f32bf16f32_compute_residential");
                ig_bgemm_f32bf16f32_compute_residential(
                        transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres);
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            // TimeLine t("ig_sgemm_f32i8f32_compute_residential");
            ig_sgemm_f32i8f32_compute_residential(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias, res, ldres);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            // TimeLine t("ig_hgemm_f32i8f32_compute_residential");
            ig_hgemm_f32i8f32_compute_residential(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias, res, ldres);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    template <typename WeiT>
    static void compute_resext(bool transA, int M, int N, int K, float alpha, const float *A, int lda,
            const WeiT *packedB, const float *scaleB, const float *zeroB, float beta, float *C, int ldc,
            const float *bias, float gamma, float *res, int ldres) {
        // FP32
        if constexpr (std::is_same_v<WeiT, float>) {
            // TimeLine t("ig_sgemm_compute_resext");
            ig_sgemm_compute_resext(transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, gamma, res, ldres);
        }

        // FP16
        else if constexpr (std::is_same_v<WeiT, float16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_FP16
            // TimeLine t("ig_sgemm_f32f16f32_compute_resext");
            ig_sgemm_f32f16f32_compute_resext(
                    transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, gamma, res, ldres);
#elif defined(AVX512_FP16_WEIGHT_ONLY_FP16)
            // TimeLine t("ig_hgemm_f32f16f32_compute_resext");
            ig_hgemm_f32f16f32_compute_resext(
                    transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, gamma, res, ldres);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_FP16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // BF16
        else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_BF16
            // TimeLine t("ig_sgemm_f32bf16f32_compute_resext");
            ig_sgemm_f32bf16f32_compute_resext(
                    transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, gamma, res, ldres);
#elif defined(AVX512_BF16_WEIGHT_ONLY_BF16)
            if (M > USE_AMX_M) {
                // TimeLine t("ig_amx_sgemm_f32bf16f32_compute_residential");
#pragma omp parallel for collapse(2)
                for (int i = 0; i < M; ++i) {
                    for (int j = 0; j < N; ++j) {
                        res[i * ldres + j] = res[i * ldres + j] * gamma;
                    }
                }
                ig_amx_sgemm_f32bf16f32_compute_residential(
                        transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, res, ldres);
            } else {
                // TimeLine t("ig_bgemm_f32bf16f32_compute_resext");
                ig_bgemm_f32bf16f32_compute_resext(
                        transA, M, N, K, alpha, A, lda, packedB, beta, C, ldc, bias, gamma, res, ldres);
            }
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_BF16 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }

        // INT8
        else if constexpr (std::is_same_v<WeiT, int8_t>) {
#ifdef AVX512_FP32_WEIGHT_ONLY_INT8
            // TimeLine t("ig_sgemm_f32i8f32_compute_resext");
            ig_sgemm_f32i8f32_compute_resext(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias, gamma, res, ldres);
#elif defined(AVX512_FP16_WEIGHT_ONLY_INT8)
            // TimeLine t("ig_hgemm_f32i8f32_compute_resext");
            ig_hgemm_f32i8f32_compute_resext(
                    transA, M, N, K, alpha, A, lda, packedB, scaleB, zeroB, beta, C, ldc, bias, gamma, res, ldres);
#else
            printf("%s:%d: Need to define WEIGHT_ONLY_INT8 kernel data type.\n", __FILE__, __LINE__);
            exit(-1);
#endif
        }
    }

    enum matmul_kinds {
        Basic = 0,
        BiasAdd = 1,
        BiasAdd_Relu = 2,
        Silu = 3,
        Resmul = 4,
        Residential = 5,
        Resext = 6,
    };

    static std::string create_key(bool transA, int M, int N, int K, int matmul_kind) {
        std::string key = std::to_string(transA) + "_" + std::to_string(M) + "_" + std::to_string(N) + "_"
                + std::to_string(K) + "_" + std::to_string(matmul_kind);
        return key;
    }


};
