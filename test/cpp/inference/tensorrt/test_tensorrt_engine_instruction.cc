/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <memory>

#include "paddle/common/ddim.h"
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/framework/new_executor/instruction/tensorrt_engine_instruction.h"
#include "paddle/fluid/framework/new_executor/interpreter/execution_config.h"
#include "paddle/fluid/framework/new_executor/standalone_executor.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_api.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/tensorrt_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/pir/include/core/builtin_dialect.h"

TEST(TensorRTEngineInstructionTest, test_tensorrt_engine_instruction) {
  // 1. Init env
  const int size = 1;
  float raw_weight[1] = {2.};  // Weight in CPU memory.
  float raw_bias[1] = {0.};
  paddle::framework::InitMemoryMethod();
  paddle::framework::InitDevices();
  paddle::framework::InitDefaultKernelSignatureMap();
  std::unique_ptr<paddle::framework::Scope> scope =
      std::make_unique<paddle::framework::Scope>();
  auto dev_ctx =
      paddle::platform::DeviceContextPool::Instance().Get(phi::GPUPlace());
  auto weight_tensor = scope->Var("weight")->GetMutable<phi::DenseTensor>();
  weight_tensor->Resize({1});
  dev_ctx->Alloc<float>(weight_tensor);
  auto y_tensor = scope->Var("y")->GetMutable<phi::DenseTensor>();
  y_tensor->Resize({1});
  dev_ctx->Alloc<float>(y_tensor);

  // 2. construct trt engine
  paddle::inference::tensorrt::TensorRTEngine::ConstructionParams params;
  params.max_batch_size = 10;
  params.max_workspace_size = 1 << 10;
  auto engine =
      std::make_unique<paddle::inference::tensorrt::TensorRTEngine>(params);
  engine->InitNetwork();

  LOG(INFO) << "create weights";
  paddle::inference::tensorrt::TensorRTEngine::Weight weight(
      nvinfer1::DataType::kFLOAT, raw_weight, size);
  paddle::inference::tensorrt::TensorRTEngine::Weight bias(
      nvinfer1::DataType::kFLOAT, raw_bias, size);
  auto *x = engine->DeclareInput(
      "x", nvinfer1::DataType::kFLOAT, nvinfer1::Dims3{1, 1, 1});
  auto *fc_layer = TRT_ENGINE_ADD_LAYER(
      engine, FullyConnected, *x, size, weight.get(), bias.get());
  PADDLE_ENFORCE_NOT_NULL(fc_layer,
                          paddle::platform::errors::InvalidArgument(
                              "TRT fully connected layer building failed."));

  engine->DeclareOutput(fc_layer, 0, "y");
  std::vector<std::string> input_names = {"x", ""};
  std::vector<std::string> output_names = {"y"};
  std::vector<int> origin_output_rank = {1};
  std::vector<phi::DataType> origin_outputs_dtype = {phi::DataType::FLOAT32};
  std::vector<paddle::dialect::IrTensor> outs_meta;
  outs_meta.emplace_back(phi::DataType::FLOAT32,
                         phi::DDim({1}),
                         phi::DataLayout::NCHW,
                         pir::LoD(),
                         0);
  LOG(INFO) << "freeze network";
  engine->FreezeNetwork();
  ASSERT_EQ(engine->engine()->getNbBindings(), 2);

  // 3. Build PIR Program
  // x --------
  //           |------> trt_op(matmul) -> pd_op.assign -> output value
  // weight ---
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::TensorRTOpDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Program program(ctx);
  pir::Builder builder(ctx, program.block());
  auto x_value =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{1}, 100.0f)
          .out();
  auto weight_value =
      builder.Build<pir::ParameterOp>("weight", x_value.type()).result(0);
  auto y_value =
      builder.Build<pir::ParameterOp>("y", x_value.type())
          .result(0);  // Use for load y, although y is not a parameter
  std::vector<pir::Value> combine_input = {x_value, weight_value};
  auto tensorrt_input = builder.Build<pir::CombineOp>(combine_input).out();
  auto tensorrt_result =
      builder
          .Build<paddle::dialect::TensorRTEngineOp>(tensorrt_input,
                                                    engine.get(),
                                                    10,
                                                    1 << 10,
                                                    false,
                                                    input_names,
                                                    output_names,
                                                    origin_output_rank,
                                                    origin_outputs_dtype,
                                                    outs_meta)
          .out();
  auto assign_input = builder.Build<pir::SplitOp>(tensorrt_result).outputs()[0];
  builder.Build<paddle::dialect::AssignOut_Op>(assign_input, y_value);
  y_value.set_attribute(
      "persistable", pir::BoolAttribute::get(pir::IrContext::Instance(), true));

  // 4. Run Program
  auto kernel_program =
      paddle::dialect::PdOpLowerToKernelPass(&program, phi::GPUPlace());
  std::unique_ptr<paddle::framework::NaiveExecutor> executor =
      std::make_unique<paddle::framework::NaiveExecutor>(phi::GPUPlace());
  paddle::framework::interpreter::ExecutionConfig execution_config;
  execution_config.create_local_scope = false;
  execution_config.used_for_inference = true;
  executor->PrepareInterpreterCore(
      scope.get(), *(kernel_program.get()), execution_config);
  executor->RunInterpreterCore();

  // check
  auto y = scope->Var("y")->Get<phi::DenseTensor>();
  phi::DenseTensor result;
  phi::Copy(*(static_cast<phi::CPUContext *>(dev_ctx)),
            y,
            phi::CPUPlace(),
            true,
            &result);
  auto *result_data = result.data<float>();
  ASSERT_EQ(result_data[0], 200);
}

TEST(TensorRTEngineInstructionTest, test_tensorrt_engine_instruction_dynamic) {
  // 1. Init env
  paddle::framework::InitMemoryMethod();
  paddle::framework::InitDevices();
  paddle::framework::InitDefaultKernelSignatureMap();
  std::unique_ptr<paddle::framework::Scope> scope =
      std::make_unique<paddle::framework::Scope>();
  auto dev_ctx =
      paddle::platform::DeviceContextPool::Instance().Get(phi::GPUPlace());
  auto y_tensor = scope->Var("y")->GetMutable<phi::DenseTensor>();
  y_tensor->Resize({8, 8, 4});
  dev_ctx->Alloc<float>(y_tensor);

  // 2. construct trt engine
  std::map<std::string, std::vector<int>> min_input_shape = {
      {"input", {1, 32}}};
  std::map<std::string, std::vector<int>> max_input_shape = {
      {"input", {18, 32}}};
  std::map<std::string, std::vector<int>> optim_input_shape = {
      {"input", {18, 32}}};
  std::map<std::string, std::vector<int>> min_input_value = {
      {"shape", {1, 8, 4}}};
  std::map<std::string, std::vector<int>> max_input_value = {
      {"shape", {18, 8, 4}}};
  std::map<std::string, std::vector<int>> optim_input_value = {
      {"shape", {18, 8, 4}}};

  paddle::inference::tensorrt::TensorRTEngine::ConstructionParams params;
  params.max_batch_size = 16;
  params.max_workspace_size = 1 << 10;
  params.with_dynamic_shape = true;
  params.min_input_shape = min_input_shape;
  params.max_input_shape = max_input_shape;
  params.optim_input_shape = optim_input_shape;
  params.min_shape_tensor = min_input_value;
  params.max_shape_tensor = max_input_value;
  params.optim_shape_tensor = optim_input_value;

  auto engine = std::make_unique<paddle::inference::tensorrt::TensorRTEngine>(
      params, paddle::inference::tensorrt::NaiveLogger::Global());
  engine->InitNetwork();

  std::cout << "with_dynamic_shape: " << engine->with_dynamic_shape()
            << std::endl;
  auto *x = engine->DeclareInput(
      "input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2{-1, 32});
  nvinfer1::Dims shape_dim;
  shape_dim.nbDims = 1;
  shape_dim.d[0] = 3;
  auto *shape =
      engine->DeclareInput("shape", nvinfer1::DataType::kINT32, shape_dim);
  auto layer = engine->network()->addShuffle(*x);
  layer->setInput(1, *shape);
  PADDLE_ENFORCE_NOT_NULL(layer,
                          paddle::platform::errors::InvalidArgument(
                              "TRT shuffle layer building failed."));
  engine->DeclareOutput(layer, 0, "y");
  engine->FreezeNetwork();
  LOG(INFO) << "freeze network";

  // 3. Build PIR Program
  // x --------
  //           |------> trt_op(matmul) -> pd_op.assign -> output value
  // weight ---
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::TensorRTOpDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Program program(ctx);
  pir::Builder builder(ctx, program.block());
  auto x_value =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{8, 32}, 1.0f)
          .out();
  auto shape_value = builder
                         .Build<paddle::dialect::FullIntArrayOp>(
                             std::vector<int64_t>({8, 8, 4}),
                             phi::DataType::INT64,
                             phi::CPUPlace())
                         .out();
  auto y_value =
      builder.Build<pir::ParameterOp>("y", x_value.type())
          .result(0);  // Use for load y, although y is not a parameter
  std::vector<pir::Value> combine_input = {x_value, shape_value};
  auto tensorrt_input = builder.Build<pir::CombineOp>(combine_input).out();

  auto vec_shape = paddle::dialect::GetInt64Vector(
      shape_value.defining_op()
          ->dyn_cast<paddle::dialect::FullIntArrayOp>()
          .attribute("value"));

  std::vector<std::string> input_names = {"input", "shape"};
  std::vector<std::string> output_names = {"y"};
  std::vector<int> origin_output_rank = {1};
  std::vector<phi::DataType> origin_outputs_dtype = {phi::DataType::FLOAT32};
  std::vector<paddle::dialect::IrTensor> outs_meta;
  outs_meta.emplace_back(phi::DataType::FLOAT32,
                         phi::DDim(vec_shape.data(), vec_shape.size()),
                         phi::DataLayout::NCHW,
                         pir::LoD(),
                         0);

  auto tensorrt_result =
      builder
          .Build<paddle::dialect::TensorRTEngineOp>(tensorrt_input,
                                                    engine.get(),
                                                    10,
                                                    1 << 10,
                                                    false,
                                                    input_names,
                                                    output_names,
                                                    origin_output_rank,
                                                    origin_outputs_dtype,
                                                    outs_meta)
          .out();
  auto assign_input = builder.Build<pir::SplitOp>(tensorrt_result).outputs()[0];
  builder.Build<paddle::dialect::AssignOut_Op>(assign_input, y_value);
  y_value.set_attribute(
      "persistable", pir::BoolAttribute::get(pir::IrContext::Instance(), true));

  // 4. Run Program
  auto kernel_program =
      paddle::dialect::PdOpLowerToKernelPass(&program, phi::GPUPlace());
  std::unique_ptr<paddle::framework::NaiveExecutor> executor =
      std::make_unique<paddle::framework::NaiveExecutor>(phi::GPUPlace());
  paddle::framework::interpreter::ExecutionConfig execution_config;
  execution_config.create_local_scope = false;
  execution_config.used_for_inference = true;
  executor->PrepareInterpreterCore(
      scope.get(), *(kernel_program.get()), execution_config);
  executor->RunInterpreterCore();

  // check
  auto y = scope->Var("y")->Get<phi::DenseTensor>();
  phi::DenseTensor result;
  phi::Copy(*(static_cast<phi::CPUContext *>(dev_ctx)),
            y,
            phi::CPUPlace(),
            true,
            &result);
  ASSERT_EQ(result.dims()[0], 8);
  ASSERT_EQ(result.dims()[1], 8);
  ASSERT_EQ(result.dims()[2], 4);
  auto *result_data = result.data<float>();
  ASSERT_EQ(result_data[0], 1);
}
