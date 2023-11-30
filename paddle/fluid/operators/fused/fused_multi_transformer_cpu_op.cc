/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <memory>
#include <string>

#include "fused_multi_transformer_cpu_op.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "xft/common/dtype.h"
#include "xft/layers/matmul_helper.h"
#include "xft/kernels/layernorm_kernels.h"
#include "xft/kernels/rmsnorm_kernels.h"

namespace paddle {
namespace operators {

class FusedMultiTransformerCPUOp : public framework::OperatorWithKernel {
 private:
  static constexpr const char *OpName = "FusedMultiTransformerINT8Op";

 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
#define CHECK_INPUT(name) \
  OP_INOUT_CHECK(ctx->HasInput(#name), "Input", #name, OpName)
#define CHECK_INPUTS(name) \
  OP_INOUT_CHECK(ctx->HasInputs(#name), "Input", #name, OpName)
#define CHECK_OUTPUT(name) \
  OP_INOUT_CHECK(ctx->HasOutput(#name), "Output", #name, OpName)
#define CHECK_OUTPUTS(name) \
  OP_INOUT_CHECK(ctx->HasOutputs(#name), "Output", #name, OpName)

    CHECK_INPUT(X);

    // attention
    CHECK_INPUTS(QKVW);
    CHECK_INPUTS(OutLinearW);

    if (ctx->HasInput("TimeStep")) {
      CHECK_INPUTS(CacheKV);
    }

    if (ctx->HasInputs("CacheKV")) {
      CHECK_OUTPUTS(CacheKVOut);
    }

    // ffn
    CHECK_INPUTS(FFN1Weight);
    CHECK_INPUTS(FFN2Weight);

    CHECK_OUTPUT(Out);

    // x: qkv's input [batch_size, seq_len, dim_embed]
    // y: qkv's weight: [3, num_head, dim_head, dim_embed]
    auto x_dim = ctx->GetInputDim("X");
    auto y_dim = ctx->GetInputsDim("QKVW")[0];
    bool trans_qkvw = ctx->Attrs().Get<bool>("trans_qkvw");
    PADDLE_ENFORCE_EQ(
        x_dim.size(),
        3,
        platform::errors::InvalidArgument("The dimensions of x must be 3"
                                          "(batch_size, seq_len, dim_embed),"
                                          "but received dimensions of"
                                          "Input is [%d]",
                                          x_dim.size()));
    PADDLE_ENFORCE_EQ(y_dim.size(),
                      4,
                      platform::errors::InvalidArgument(
                          "The dimensions of qkv_weight must be 4"
                          "(3, num_head, dim_head, dim_embed),"
                          "but received dimensions of"
                          "Input is [%d]",
                          y_dim.size()));
    PADDLE_ENFORCE_EQ(
        x_dim[2],
        trans_qkvw ? y_dim[3] : y_dim[0],
        platform::errors::InvalidArgument(
            "ShapeError: the dimension of x_dim[2] and y_dim[3](trans_qkvw is "
            "true) or y_dim[0](trans_qkvw is false)"
            "must be equal. But received: the shape "
            "of input x = [%s], and the shape of "
            "input qkv_weight = [%s]",
            x_dim,
            y_dim));

    if (ctx->Attrs().Get<int>("ring_id") == -1) {
      if (trans_qkvw) {
        PADDLE_ENFORCE_EQ(y_dim[1] * y_dim[2],
                          y_dim[3],
                          platform::errors::InvalidArgument(
                              "The dimensions of qkv_weight must be 4"
                              "(3, num_head, dim_head, dim_embed),"
                              "and must satisfy the limitations: "
                              "(num_head * dim_head == dim_embed)"));

      } else {
        PADDLE_ENFORCE_EQ(y_dim[2] * y_dim[3],
                          y_dim[0],
                          platform::errors::InvalidArgument(
                              "The dimensions of qkv_weight must be 4"
                              "(dim_embed, 3, num_head, dim_head),"
                              "and must satisfy the limitations: "
                              "(num_head * dim_head == dim_embed)"));
      }
    }

    if (ctx->HasInputs("CacheKV")) {
      // [2, batch_size, num_head, max_seq_len, head_size]
      const auto &c_dims = ctx->GetInputsDim("CacheKV");
      const auto &c_dim = c_dims[0];

      PADDLE_ENFORCE_EQ(
          c_dim.size(),
          5,
          paddle::platform::errors::InvalidArgument(
              "The CacheKV must be 5 dims, but got %d", c_dim.size()));
      PADDLE_ENFORCE_EQ(c_dim[0],
                        2,
                        paddle::platform::errors::InvalidArgument(
                            "The first dim of CacheKV must be 2, but got %d",
                            c_dim[0]));  // 2
      PADDLE_ENFORCE_EQ(c_dim[1],
                        x_dim[0],
                        paddle::platform::errors::InvalidArgument(
                            "The second dim of CacheKV must be equal with "
                            "batch size %d, but got %d",
                            x_dim[0],
                            c_dim[1]));  // batch_size
      PADDLE_ENFORCE_EQ(c_dim[2],
                        trans_qkvw ? y_dim[1] : y_dim[2],
                        paddle::platform::errors::InvalidArgument(
                            "The third dim of CacheKV must be equal with num "
                            "head %d, but got %d",
                            trans_qkvw ? y_dim[1] : y_dim[2],
                            c_dim[2]));  // num_head
      PADDLE_ENFORCE_GT(
          c_dim[3],
          0,
          paddle::platform::errors::InvalidArgument(
              "The forth dim of CacheKV must be greater than 0, but got %d",
              c_dim[3]));  // cache_seq_len
      PADDLE_ENFORCE_EQ(c_dim[4],
                        trans_qkvw ? y_dim[2] : y_dim[3],
                        paddle::platform::errors::InvalidArgument(
                            "The fifth dim of CacheKV must be equal with head "
                            "size %d, but got %d",
                            trans_qkvw ? y_dim[2] : y_dim[3],
                            c_dim[4]));  // head_size
    }

    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                          ctx.GetPlace());
  }

  phi::KernelKey GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DenseTensor &tensor,
      const phi::KernelKey &expected_kernel_type) const override {
    if (var_name == "TimeStep") {
      VLOG(10) << "var_name:" << var_name << " need not to transform";
      return phi::KernelKey(phi::Backend::ALL_BACKEND,
                            expected_kernel_type.layout(),
                            expected_kernel_type.dtype());
    }
    return phi::KernelKey(
        tensor.place(), tensor.layout(), expected_kernel_type.dtype());
  }
};

class FusedMultiTransformerCPUOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor.");
    AddInput("LnScale",
             "Scale is a 1-dimensional tensor of size "
             "H. Here, H represents the last dimension of its input tensor.")
        .AsDuplicable();
    AddInput("LnBias",
             "Bias is a 1-dimensional tensor of size "
             "H. Here, H represents the last dimension of its input tensor.")
        .AsDuplicable();
    AddInput("QKVW", "The qkv weight tensor.").AsDuplicable();
    AddInput("QKVBias", "The qkv bias tensor.").AsDispensable().AsDuplicable();

    AddInput("CacheKV", "(optional) The cached KV for generation inference.")
        .AsDispensable()
        .AsDuplicable();
    AddInput("TimeStep",
             "(optional, int) The time step for generation inference.")
        .AsDispensable();
    AddInput("SrcMask", "(optional) The attention mask tensor in fmha.")
        .AsDispensable();
    AddInput("OutLinearW", "The out_linear weight tensor.").AsDuplicable();
    AddInput("OutLinearBias", "The out_linear bias tensor.")
        .AsDispensable()
        .AsDuplicable();

    AddInput("FFNLnScale", "The layer_norm scale of FusedFeedForward op")
        .AsDuplicable();
    AddInput("FFNLnBias", "The layer_norm bias of FusedFeedForward op")
        .AsDuplicable();

    AddInput("FFN1Weight", "The linear1 weight of FusedFeedForward op")
        .AsDuplicable();
    AddInput("FFN1Bias", "The linear1 bias of FusedFeedForward op")
        .AsDispensable()
        .AsDuplicable();

    AddInput("FFN2Weight", "The linear2 weight of FusedFeedForward op")
        .AsDuplicable();
    AddInput("FFN2Bias", "The linear2 bias input of FusedFeedForward op")
        .AsDispensable()
        .AsDuplicable();

    AddOutput("CacheKVOut", "The updated cache KV. Inplace with CacheKV")
        .AsDispensable()
        .AsDuplicable();
    AddOutput("Out", "Result after multi .");

    AddAttr<bool>("pre_layer_norm",
                  "if true, the attention op uses pre_layer_norm architecure, "
                  "else, uses post_layer_norm architecuture. "
                  "[default true].")
        .SetDefault(true);
    AddAttr<float>("epsilon",
                   "Constant for numerical stability [default 1e-5].")
        .SetDefault(1e-5)
        .AddCustomChecker([](const float &epsilon) {
          PADDLE_ENFORCE_EQ(epsilon >= 0.0f && epsilon <= 0.001f,
                            true,
                            platform::errors::InvalidArgument(
                                "'epsilon' in Op(LayerNorm) should be between"
                                "0.0 and 0.001, But received [%s].",
                                epsilon));
        });

    AddAttr<std::string>("act_method", "act_method").SetDefault("gelu");
    AddAttr<bool>(
        "trans_qkvw",
        "Whether the weights of qkv should be transposed. If true,"
        "the shape eights of qkv should be [3, num_head, dim_head, dim_embed]."
        "Otherwise the shape of weights of qkv should be"
        "[dim_embed, 3, num_head, dim_head]")
        .SetDefault(true);

    AddAttr<int>(
        "ring_id",
        "ring id for tensor model parallel. distributed training and inference")
        .SetDefault(-1);

    AddAttr<int>("num_head", "num_head").SetDefault(0);
    AddAttr<int>("dim_head", "dim_head").SetDefault(0);
    AddAttr<int>("dim_ffn", "dim_ffn").SetDefault(0);

    AddComment(R"DOC(fused multi transformer cpu layers op)DOC");
  }
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"

template <typename T, typename DeviceContext>
class FusedMultiTransformerCPUOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto& device_ctx = ctx.device_context();

    auto *input_x = ctx.Input<phi::DenseTensor>("X");
    auto qkv_weights = ctx.MultiInput<phi::DenseTensor>("QKVW");
    auto qkv_biases = ctx.Input<phi::DenseTensor>("QKVBias");
    int num_layer = qkv_weights.size();
    PADDLE_ENFORCE_GT(num_layer, 0, "num_layer should be greater than 0");

    auto out_linear_weights = ctx.MultiInput<phi::DenseTensor>("OutLinearW");
    auto out_linear_biases = ctx.MultiInput<phi::DenseTensor>("OutLinearBias");

    const auto pre_layer_norm = ctx.Attr<bool>("pre_layer_norm");
    const float epsilon = ctx.Attr<float>("epsilon");
    auto ln_scales = ctx.MultiInput<phi::DenseTensor>("LnScale");
    auto ln_biases = ctx.MultiInput<phi::DenseTensor>("LnBias");

    auto *time_step = ctx.Input<phi::DenseTensor>("TimeStep");

    const bool trans_qkvw = ctx.Attr<bool>("trans_qkvw");

    const auto input_x_dims = input_x->dims();
    int bsz = input_x_dims[0];
    int seq_len = input_x_dims[1];
    int dim_embed = input_x_dims[2];
    int bsz_seq = bsz * seq_len;

    const auto qkv_w_dims = qkv_weights[0]->dims();
    int num_head = trans_qkvw ? qkv_w_dims[1] : qkv_w_dims[2];
    int dim_head = trans_qkvw ? qkv_w_dims[2] : qkv_w_dims[3];
    int hidden_size = num_head * dim_head;
    int output_size = 3 * hidden_size;
    int input_size = dim_embed;

    phi::DenseTensor qkv_out;
    qkv_out.Resize({{bsz, seq_len, 3, num_head, dim_head}});
    auto *qkv_out_data = device_ctx.Alloc<T>(&qkv_out, qkv_out.numel() * sizeof(T));

    phi::DenseTensor layernorm_out;
    layernorm_out.Resize({{bsz, seq_len, dim_embed}});
    device_ctx.Alloc<float>(&layernorm_out, layernorm_out.numel() * sizeof(float));
    auto* layernorm_out_data = layernorm_out.data<float>();

    auto* input_x_data = input_x->data<T>();
    auto *out = ctx.Output<phi::DenseTensor>("Out");
    auto *out_data = device_ctx.Alloc<T>(out, out->numel() * sizeof(T));

    for(int i=0; i< num_layer; ++i){
      auto *ln_scale_data = ln_scales[i]->data<float>();
      bool compute_ln_bias = ln_biases.size() > 0;
      auto *ln_bias_data = compute_ln_bias? ln_biases[i]->data<float>() : nullptr;
      if (i == 0 && pre_layer_norm) {
        if(!compute_ln_bias){
          xft::invokeRmsNorm(out_data, input_x_data, ln_scale_data, bsz_seq, dim_embed, -1, -1, 1e-5f);
        }
      }
      
      // auto qkv_weight = qkv_weights[i];
      // auto qkv_bias = qkv_biases[i];
    }
  }
};

#pragma GCC diagnostic pop

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    fused_multi_transformer_cpu,
    ops::FusedMultiTransformerCPUOp,
    ops::FusedMultiTransformerCPUOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);


namespace ops = paddle::operators;
namespace plat = paddle::platform;
PD_REGISTER_STRUCT_KERNEL(fused_multi_transformer_cpu,
                          CPU,
                          ALL_LAYOUT,
                          ops::FusedMultiTransformerCPUOpKernel,
                          float) {}
