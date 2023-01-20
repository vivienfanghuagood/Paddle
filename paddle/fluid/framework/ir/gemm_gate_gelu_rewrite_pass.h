// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>
#include <string>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"

#define GET_IR_NODE(node__) \
  GET_IR_NODE_FROM_SUBGRAPH(node__, node__, gemm_split_gelu_mul_pattern);
#define GET_NODES               \
  GET_IR_NODE(mul);       \
  GET_IR_NODE(mul_weight);       \
  GET_IR_NODE(mul_out);       \
  GET_IR_NODE(elementwise_add); \
  GET_IR_NODE(bias); \
  GET_IR_NODE(elementwise_add_out); \
  GET_IR_NODE(split);           \
  GET_IR_NODE(split_out0);      \
  GET_IR_NODE(split_out1);      \
  GET_IR_NODE(gelu);            \
  GET_IR_NODE(gelu_out);        \
  GET_IR_NODE(elementwise_mul); \
  GET_IR_NODE(elementwise_mul_out);

namespace paddle {
namespace framework {
namespace ir {

class Node;

namespace patterns {

// BEGINNOLINT
struct GemmGateGeluPattern : public PatternBase {
  GemmGateGeluPattern(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "gemm_gate_gelu") {}

  PDNode* operator()(PDNode *x){
    x->assert_is_op_input("mul", "X");
    auto *mul_op = pattern->NewNode(mul_repr())->assert_is_op("mul");
    auto *mul_w_var = pattern->NewNode(mul_weight_repr())
                        ->AsInput()
                        ->assert_is_persistable_var()
                        ->assert_is_op_input("mul", "Y");

    auto *mul_out = pattern->NewNode(mul_out_repr())->AsOutput()->assert_is_op_output("mul");
    mul_op->LinksFrom({x, mul_w_var}).LinksTo({mul_out});

    mul_out->AsIntermediate()->assert_is_op_input("elementwise_add");
    auto *elementwise_add_op = pattern->NewNode(elementwise_add_repr())->assert_is_op("elementwise_add");
    auto *bias = pattern->NewNode(bias_repr())
                     ->assert_is_op_input("elementwise_add")
                     ->assert_is_persistable_var()
                     ->AsInput();
    auto *elementwise_add_out = pattern->NewNode(elementwise_add_out_repr())->AsOutput()->assert_is_op_output("elementwise_add");
    elementwise_add_op->LinksFrom({mul_out, bias}).LinksTo({elementwise_add_out});

    auto *split_op = pattern->NewNode(split_repr())->assert_is_op("split");
    auto *split_out0 = pattern->NewNode(split_out0_repr())
                          ->AsOutput()
                          ->assert_is_op_nth_output("split", "Out", 0);
    auto *split_out1 = pattern->NewNode(split_out1_repr())
                          ->AsOutput()
                          ->assert_is_op_nth_output("split", "Out", 1);
    // Add links for split op.
    split_op->LinksFrom({elementwise_add_out}).LinksTo({split_out0, split_out1});

    // Create nodes for gelu op.
    split_out1->assert_is_op_input("gelu");
    auto *gelu = pattern->NewNode(gelu_repr())->assert_is_op("gelu");

    auto *gelu_out = pattern->NewNode(gelu_out_repr())
                        ->AsOutput()
                        ->assert_is_op_output("gelu", "Out");

    // Add links for gelu op.
    gelu->LinksFrom({split_out1}).LinksTo({gelu_out});

    // Create nodes for elementwise_mul op.
    gelu_out->assert_is_op_input("elementwise_mul");
    split_out0->assert_is_op_input("elementwise_mul");
    auto *elementwise_mul =
        pattern->NewNode(elementwise_mul_repr())->assert_is_op("elementwise_mul");

    auto *elementwise_mul_out =
        pattern->NewNode(elementwise_mul_out_repr())
            ->AsOutput()
            ->assert_is_op_output("elementwise_mul", "Out");

    // Add links for elementwise_mul op.
    elementwise_mul->LinksFrom({gelu_out, split_out0})
        .LinksTo({elementwise_mul_out});
    return elementwise_mul_out;
  }

  // declare operator node's name
  PATTERN_DECL_NODE(mul);
  PATTERN_DECL_NODE(mul_weight);
  PATTERN_DECL_NODE(mul_out);
  PATTERN_DECL_NODE(elementwise_add);
  PATTERN_DECL_NODE(bias);
  PATTERN_DECL_NODE(elementwise_add_out);
  PATTERN_DECL_NODE(split);
  PATTERN_DECL_NODE(split_out0);
  PATTERN_DECL_NODE(split_out1);
  PATTERN_DECL_NODE(gelu);
  PATTERN_DECL_NODE(gelu_out);
  PATTERN_DECL_NODE(elementwise_mul);
  PATTERN_DECL_NODE(elementwise_mul_out);
};

};  // namespace patterns

//      mul
//       |
// elementwise_add
//       |
//     split
//   |       |
//   |     gelu
//   |       |
// elementwise_mul     ->      mul_add_mul( mul_add_gelu(x) )
//       |
//

class MulSplitGeluMulFusePass : public FusePassBase {

public:
  MulSplitGeluMulFusePass(){
    AddOpCompat(OpCompat("mul"))
        .AddInput("X")
        .IsTensor()
        .End()
        .AddOutput("Out")
        .IsTensor()
        .End()
        .AddAttr("axis")
        .IsNumEQ(2)
        .End()
        .AddAttr("num")
        .IsNumEQ(2)
        .End();

    AddOpCompat(OpCompat("elementwise_add"))
        .AddInput("X")
        .IsTensor()
        .End()
        .AddInput("Y")
        .IsTensor()
        .End()
        .AddOutput("Out")
        .IsTensor()
        .End()
        .AddAttr("axis")
        .IsNumEQ(2)
        .End();
    
    AddOpCompat(OpCompat("split"))
        .AddInput("X")
        .IsTensor()
        .End()
        .AddOutput("Out")
        .End()
        .AddAttr("axis")
        .IsNumEQ(2)
        .End()
        .AddAttr("num")
        .IsNumEQ(2)
        .End();

    AddOpCompat(OpCompat("gelu"))
        .AddInput("X")
        .IsTensor()
        .End()
        .AddOutput("Out")
        .IsTensor()
        .End();
    AddOpCompat(OpCompat("elementwise_mul"))
        .AddInput("X")
        .IsTensor()
        .End()
        .AddInput("Y")
        .IsTensor()
        .End()
        .AddOutput("Out")
        .IsTensor()
        .End()
        .AddAttr("axis")
        .IsIntIn({0, -1})
        .End();
  }

  virtual ~MulSplitGeluMulFusePass() {}

protected:
  void ApplyImpl(ir::Graph *graph) const override{
    int fuse_count = ApplyPattern(graph);
    AddStatis(fuse_count);
  }
  int ApplyPattern(ir::Graph *graph) const{
    PADDLE_ENFORCE_NOT_NULL(
        graph,
        platform::errors::InvalidArgument(
            "The input graph of SplitGeluMulFusePass should not be "
            "nullptr."));
    GraphPatternDetector gpd;
    FusePassBase::Init(scope_name_, graph);
    PDNode *x = gpd.mutable_pattern()
                    ->NewNode("x")
                    ->assert_is_op_input("mul", "X")
                    ->AsInput();
    patterns::GemmGateGeluPattern gemm_split_gelu_mul_pattern(gpd.mutable_pattern(), scope_name_);
    gemm_split_gelu_mul_pattern(x);
    int fuse_count = 0;
    auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                       Graph *g) {
      if (!IsCompat(subgraph, g)) {
        LOG(WARNING) << "gemm_split_gelu_mul in op compat failed.";
        return;
      }
      GET_NODES;

      // TODO change split
      
      LOG(INFO) << "!!! handle gemm gate gelu match!";

      VLOG(4) << "handle gemm_split_gelu fuse";
      // GET_IR_NODE_FROM_SUBGRAPH(mul_weight, mul_weight, gemm_split_gelu_mul_pattern);
      // GET_IR_NODE_FROM_SUBGRAPH(bias, bias, gemm_split_gelu_mul_pattern);
      // GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_out, elementwise_add_out, gemm_split_gelu_mul_pattern);
      // GET_IR_NODE_FROM_SUBGRAPH(mul, mul, gemm_split_gelu_mul_pattern);
      // GET_IR_NODE_FROM_SUBGRAPH(elementwise_add, elementwise_add, gemm_split_gelu_mul_pattern);
      // GET_IR_NODE_FROM_SUBGRAPH(mul_out, mul_out, gemm_split_gelu_mul_pattern);

      std::vector<int64_t> w_shape = mul_weight->Var()->GetShape();

      auto* scope = param_scope();
      auto* weight = scope->FindVar(mul_weight->Name())->GetMutable<phi::DenseTensor>();
      auto* bias_weight = scope->FindVar(bias->Name())->GetMutable<phi::DenseTensor>();
      auto* weight_data = weight->data<float>();
      auto* bias_data = bias_weight->data<float>();
      auto weight_dims = weight->dims();
      int weight_num = product(weight_dims);
      int half_weight_num = weight_num / 2;
      int w_h = weight_dims[0];
      int w_w = weight_dims[1];

      const std::string& w_1_name = patterns::UniqueKey(mul_weight->Name() + "_1");
      auto* w_1_var = scope->Var(w_1_name);
      auto* w_1_tensor = w_1_var->GetMutable<phi::DenseTensor>();
      w_1_tensor->Resize(DDim{w_h, w_w / 2});
      auto* w1_data_new = w_1_tensor->mutable_data<float>(platform::CPUPlace());
      memcpy(w1_data_new, weight_data, half_weight_num * sizeof(float));

      const std::string& w_2_name = patterns::UniqueKey(mul_weight->Name() + "_2");
      auto* w_2_var = scope->Var(w_2_name);
      auto* w_2_tensor = w_2_var->GetMutable<phi::DenseTensor>();
      w_2_tensor->Resize(DDim{w_h, w_w / 2});
      auto* w2_data_new = w_2_tensor->mutable_data<float>(platform::CPUPlace());
      memcpy(w2_data_new, weight_data + half_weight_num, half_weight_num * sizeof(float));

      const std::string& bias_1_name = patterns::UniqueKey(bias->Name() + "_1");
      auto* bias_1_var = scope->Var(bias_1_name);
      auto* bias_1_tensor = bias_1_var->GetMutable<phi::DenseTensor>();
      bias_1_tensor->Resize(DDim{w_w / 2});
      auto* bias_1_data_new = bias_1_tensor->mutable_data<float>(platform::CPUPlace());
      memcpy(bias_1_data_new, bias_data, w_w / 2 * sizeof(float));

      const std::string& bias_2_name = patterns::UniqueKey(bias->Name() + "_2");
      auto* bias_2_var = scope->Var(bias_2_name);
      auto* bias_2_tensor = bias_2_var->GetMutable<phi::DenseTensor>();
      bias_2_tensor->Resize(DDim{w_w / 2});
      auto* bias_2_data_new = bias_2_tensor->mutable_data<float>(platform::CPUPlace());
      memcpy(bias_2_data_new, bias_data, w_w / 2 * sizeof(float));

      const std::string& fc1_out_name = patterns::UniqueKey(elementwise_add_out->Name() + "_1");
      auto* fc1_out_var = scope->Var(fc1_out_name);
      auto* fc1_out_tensor = fc1_out_var->GetMutable<phi::DenseTensor>();
      fc1_out_tensor->Resize(DDim{w_h, w_w / 2});

      const std::string& fc2_out_name = patterns::UniqueKey(elementwise_add_out->Name() + "_2");
      auto* fc2_out_var = scope->Var(fc2_out_name);
      auto* fc2_out_tensor = fc2_out_var->GetMutable<phi::DenseTensor>();
      fc2_out_tensor->Resize(DDim{w_h, w_w / 2});

      const std::string& gelu_out_name = patterns::UniqueKey(gelu_out->Name() + "_0");
      auto* gelu_out_var = scope->Var(gelu_out_name);
      auto* gelu_out_tensor = gelu_out_var->GetMutable<phi::DenseTensor>();
      gelu_out_tensor->Resize(DDim{w_h, w_w / 2});

      OpDesc fc_1_desc;
      fc_1_desc.SetType("fc");
      // Set inputs of fc
      fc_1_desc.SetInput("Input", {subgraph.at(x)->Name()});
      fc_1_desc.SetInput("W", {w_1_name});
      fc_1_desc.SetInput("Bias", {bias_1_name});
      fc_1_desc.SetOutput("Out", {fc1_out_name});
      fc_1_desc.Flush();
      auto fc_1_node = graph->CreateOpNode(&fc_1_desc);

      OpDesc gelu_desc;
      gelu_desc.SetType("gelu");
      gelu_desc.SetInput("X", {fc1_out_name});
      gelu_desc.SetOutput("Out", {gelu_out_name});
      gelu_desc.SetAttr("approximate", false);
      gelu_desc.Flush();
      auto gelu_node = graph->CreateOpNode(&gelu_desc);

      OpDesc fc_2_desc;
      fc_2_desc.SetType("fc");
      // Set inputs of fc
      fc_2_desc.SetInput("Input", {fc1_out_name});
      fc_2_desc.SetInput("W", {w_2_name});
      fc_2_desc.SetInput("Bias", {bias_2_name});
      fc_2_desc.SetOutput("Out", {fc2_out_name});
      fc_2_desc.Flush();
      auto fc_2_node = graph->CreateOpNode(&fc_2_desc);

      OpDesc elementwise_mul_desc;
      elementwise_mul_desc.SetType("elementwise_mul");
      elementwise_mul_desc.SetInput("X", {fc1_out_name});
      elementwise_mul_desc.SetInput("Y", {gelu_out_name});
      elementwise_mul_desc.SetOutput("Out", {elementwise_mul_out->Name()});
      elementwise_mul_desc.Flush();
      auto elementwise_mul_node = graph->CreateOpNode(&elementwise_mul_desc);

      IR_NODE_LINK_TO(subgraph.at(x), fc_1_node);
      IR_NODE_LINK_TO(fc_1_node, gelu_node);
      IR_NODE_LINK_TO(gelu_node, fc_2_node);
      IR_NODE_LINK_TO(fc_2_node, elementwise_mul_node);
      IR_NODE_LINK_TO(elementwise_mul_node, elementwise_mul_out);

      std::unordered_set<const Node *> del_node_set = {
        mul, mul_out, elementwise_add, elementwise_add_out, split, split_out0, split_out1, gelu, gelu_out, elementwise_mul};
      GraphSafeRemoveNodes(graph, del_node_set);
      
      // OpDesc split_gelu_desc;
      // split_gelu_desc.SetType("split_gelu");
      // split_gelu_desc.SetInput("X", {subgraph.at(x)->Name()});
      // split_gelu_desc.SetOutput("Out", {elementwise_mul_out->Name()});
      // split_gelu_desc.Flush();
      // auto split_gelu_node = graph->CreateOpNode(&split_gelu_desc);
      // IR_NODE_LINK_TO(subgraph.at(x), split_gelu_node);
      // IR_NODE_LINK_TO(split_gelu_node, elementwise_mul_out);
      // GraphSafeRemoveNodes(graph, del_node_set);
      ++fuse_count;
    };
    gpd(graph, handler);
    return fuse_count;
  }

private:
  const std::string scope_name_{"mul_split_gelu_mul_fuse"};
};

// ENDNOLINT
}  // namespace ir
}  // namespace framework
}  // namespace paddle
