// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/transforms/sub_graph_extract_pass.h"

#include <queue>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>

#include "paddle/common/flags.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

#include "paddle/cinn/hlir/framework/pir/utils.h"

#include "paddle/fluid/pir/transforms/sub_graph_detector.h"

COMMON_DECLARE_int32(trt_min_group_size);

namespace {
using GroupOpsVec = std::vector<pir::Operation*>;

bool IsSupportedByTRT(const pir::Operation& op) {
  return op->attribute<pir::BoolAttribute>(paddle::dialect::kCanRunTrtAttr).data();
}

class TrtSubGraphExtractPass : public pir::Pass {
 public:
  SubGraphExtractPass()
      : pir::Pass("trt_sub_graph_extract_pass", 1) {}

  void Run(pir::Operation* op) override {
    auto module_op = op->dyn_cast<pir::ModuleOp>();
    PADDLE_ENFORCE_NOT_NULL(
        module_op,
        phi::errors::InvalidArgument(
            "sub_graph_extract_pass should run on module op."));
    auto& block = module_op.block();

    std::vector<GroupOpsVec> groups =
        ::pir::SubgraphDetector(&block, IsSupportedByTRT)();
    AddStatistics(groups.size());
    for (auto& group_ops : groups) {
      if(group_ops.size() < trt_min_group_size) {
        VLOG(4) << "current group_ops.size(): " << group_ops.size() << ", will fallback to paddle original graph";
        continue;
      }
      VLOG(4) << "current group_ops.size(): " << group_ops.size() << ", will lower to TensorRT graph";
      ::pir::ReplaceWithGroupOp(&block, group_ops);
    }
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }
};
}  // namespace

namespace pir {

std::unique_ptr<Pass> TrtSubGraphExtractPass() {
  return std::make_unique<TrtSubGraphExtractPass>();
}

}  // namespace pir

REGISTER_IR_PASS(trt_sub_graph_extract_pass, TrtSubGraphExtractPass);
