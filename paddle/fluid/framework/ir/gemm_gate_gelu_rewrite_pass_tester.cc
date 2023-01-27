/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>

#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/framework/ir/gemm_gate_gelu_rewrite_pass.h"

// BEGINNOLINT
namespace paddle {
namespace framework {
class VarDesc;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {

void TestMain() {

    Layers layers;
    VarDesc* x = layers.data("x");
    VarDesc* weight = layers.data("weight_0", {320, 2560}, true);
    VarDesc* bias = layers.data("bias_0", {2560}, true);
    VarDesc* mul_out = layers.mul(x, weight);
    VarDesc* add_bias_out = layers.elementwise_add(mul_out, bias);
    auto split_outs = layers.split(add_bias_out, 2);
    VarDesc* gelu_out = layers.gelu(split_outs[0]);
    VarDesc* elementwise_mul_out = layers.elementwise_mul(split_outs[1], gelu_out);

    std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
    auto pass = PassRegistry::Instance().Get("gemm_gate_gelu_rewrite_pass");
    int num_nodes_before = graph->Nodes().size();
    int num_gelu_nodes_before = GetNumOpNodes(graph, "gelu");
    VLOG(3) << DebugString(graph);

    graph.reset(pass->Apply(graph.release()));
    int num_nodes_after = graph->Nodes().size();
    int num_fused_nodes_after = GetNumOpNodes(graph, "fc");
    VLOG(3) << DebugString(graph);

    // Delete (num_gelu_nodes_before - 1) fc ops
    PADDLE_ENFORCE_EQ(
        num_fused_nodes_after,
        2,
        platform::errors::InvalidArgument("Invalid fusion group"));
}

TEST(MulSplitGeluMulFusePass, test) { TestMain(); }

}  // namespace ir
}  // namespace framework
}  // namespace paddle

// ENDNOLINT

USE_PASS(gemm_gate_gelu_rewrite_pass);
