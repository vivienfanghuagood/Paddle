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

#include <immintrin.h>

#include "../common/bfloat16.h"
#include "../common/dtype.h"
#include "../common/float16.h"
#include "../common/my_types.h"

namespace xft {

template <typename T>
struct LayerNormWeight {
    const T *gamma = nullptr;
    const T *beta = nullptr;
};

void invokeLayerNorm(float *output, const float *input, const float *gamma, const float *beta, int rows,
        int cols, int iStride = -1, int oStride = -1, float epsilon = 1e-5);

void invokeLayerNorm(float16_t *output, const float16_t *input, const float16_t *gamma, const float16_t *beta,
        int rows, int cols, int iStride = -1, int oStride = -1, float epsilon = 1e-5);

void invokeLayerNorm(bfloat16_t *output, const bfloat16_t *input, const bfloat16_t *gamma, const bfloat16_t *beta,
        int rows, int cols, int iStride = -1, int oStride = -1, float epsilon = 1e-5);

} // namespace xft