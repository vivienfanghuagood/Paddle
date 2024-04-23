#include "paddle/fluid/pir/dialect/operator/ir/tensorrt_engine_op.h"


namespace paddle {
namespace dialect {
void TensorRTEngineOp::Build(pir::Builder &builder,             // NOLINT
                      pir::OperationArgument &argument,  // NOLINT
                      pir::Value combined_inputs,
                      std::vector<pir::Type> &&output_types) {
  argument.AddInput(combined_inputs);
  argument.output_types.swap(output_types);
  argument.AddRegion().emplace_back();
}

pir::Block & TensorRTEngineOp::forward_block() {
  auto &region = forward_region();
  if (region.empty()) {
    region.emplace_back();
  }

  return region.front();
}


}

