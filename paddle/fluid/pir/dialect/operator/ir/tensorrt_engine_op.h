#pragma once
#include <vector>

#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/op_base.h"

namespace paddle {
namespace dialect {

class TensorRTEngineOp: public pir::Op<TensorRTEngineOp> {
 public:
  using Op::Op;
  static const char *name() { return "pd_op.tensorrtengine"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  TEST_API static void Build(pir::Builder &builder,             // NOLINT
                             pir::OperationArgument &argument,  // NOLINT
                             const std::vector<pir::Value> &inputs);
  TEST_API pir::Block &forward_block();

  pir::Region &forward_region() { return (*this)->region(0); }
  
  void Print(pir::IrPrinter &printer){}  // NOLINT
  void VerifySig(){}
  void VerifyRegion(){}

  void UpdateOutput(){}
};

}
}

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::TensorRTEngineOp);