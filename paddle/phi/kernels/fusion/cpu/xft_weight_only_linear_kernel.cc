/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

#include "xft/utils/matmul_helper.h"
// #include "xft/common/myt"

namespace phi {
namespace fusion{

template <typename T, typename Context>
void FusedWeightOnlyLinearKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& weight,
                            const paddle::optional<DenseTensor>& bias,
                            const DenseTensor& weight_scale,
                            const DenseTensor& weight_zero_point,
                            const std::string& weight_dtype,
                            DenseTensor* out) {
    dev_ctx.template Alloc<T>(out);
    const T* x_data = x.data<T>();
    const int8_t* weight_data = weight.data<int8_t>();
    const T* bias_data = bias ? bias.get().data<T>() : nullptr;
    const T* weight_scale_data = weight_scale.data<T>();
    const T* zero_point_data = weight_zero_point.data<T>();
    T* out_data = out->data<T>();
    // const auto x_dims = x.dims();
    const auto w_dims = weight.dims();
    int n = weight_scale.dims()[0];
    int k = w_dims[1];
    int m = x.numel() / k;

    if(bias_data){
        // if bias is not null, then call `compute_bias` function
        
        MMHelper::compute_bias<float, int8_t, float>(false, m, n, k, 1.0f, x_data, k, weight_data, weight_scale_data, zero_point_data, 0.0, out_data, n, bias_data);
    }
    else{
        MMHelper::compute<float, int8_t, float>(false, m, n, k, 1.0f, x_data, k, weight_data, weight_scale_data, zero_point_data, 0.0, out_data, n);
    }
}

}
}  // namespace phi

PD_REGISTER_KERNEL(fused_weight_only_linear,
                   CPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedWeightOnlyLinearKernel,
                   float) {}

