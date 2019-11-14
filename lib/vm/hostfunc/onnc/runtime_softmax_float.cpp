#include "vm/hostfunc/onnc/runtime_softmax_float.h"
#include "executor/common.h"
#include "executor/worker/util.h"
#include "onnc/onnc_runtime.h"

#include <stdbool.h>
#include <stdint.h>

namespace SSVM {
namespace Executor {

ONNCRuntimeSoftmaxFloat::ONNCRuntimeSoftmaxFloat() {
  appendParamDef(AST::ValType::I32);
  appendParamDef(AST::ValType::I32);
  appendParamDef(AST::ValType::I32);
  appendParamDef(AST::ValType::I32);
  appendParamDef(AST::ValType::I32);
  appendParamDef(AST::ValType::I32);
  appendParamDef(AST::ValType::I32);
  appendParamDef(AST::ValType::I32);
}

ErrCode
ONNCRuntimeSoftmaxFloat::run(std::vector<std::unique_ptr<ValueEntry>> &Args,
                             std::vector<std::unique_ptr<ValueEntry>> &Res,
                             StoreManager &Store,
                             Instance::ModuleInstance *ModInst) {
  /// Arg: void* onnc_runtime_context,
  ///      const float *input_input,
  ///      int32_t input_input_ndim,
  ///      const int32_t *input_input_dims,
  ///      float *output_output,
  ///      int32_t output_output_ndim,
  ///      const int32_t *output_output_dims,
  ///      int32_t axis
  if (Args.size() != 8) {
    return ErrCode::CallFunctionError;
  }
  ErrCode Status = ErrCode::Success;
  unsigned int RuntimeContextPtr = retrieveValue<uint32_t>(*Args[7].get());
  unsigned int InPtr = retrieveValue<uint32_t>(*Args[6].get());
  unsigned int InNDim = retrieveValue<uint32_t>(*Args[5].get());
  unsigned int InDimsPtr = retrieveValue<uint32_t>(*Args[4].get());
  unsigned int OutPtr = retrieveValue<uint32_t>(*Args[3].get());
  unsigned int OutNDim = retrieveValue<uint32_t>(*Args[2].get());
  unsigned int OutDimsPtr = retrieveValue<uint32_t>(*Args[1].get());
  unsigned int Axis = retrieveValue<uint32_t>(*Args[0].get());

  /// Get memory instance.
  unsigned int MemoryAddr = 0;
  Instance::MemoryInstance *MemInst = nullptr;
  if ((Status = ModInst->getMemAddr(0, MemoryAddr)) != ErrCode::Success) {
    return Status;
  }
  if ((Status = Store.getMemory(MemoryAddr, MemInst)) != ErrCode::Success) {
    return Status;
  }

  void *RuntimeContext =
      reinterpret_cast<void *>(MemInst->getPointer(RuntimeContextPtr));
  int32_t *InDims = reinterpret_cast<int32_t *>(MemInst->getPointer(InDimsPtr));
  int32_t *OutDims =
      reinterpret_cast<int32_t *>(MemInst->getPointer(OutDimsPtr));
  float *In = reinterpret_cast<float *>(MemInst->getPointer(InPtr));
  float *Out = reinterpret_cast<float *>(MemInst->getPointer(OutPtr));

  ONNC_RUNTIME_softmax_float(RuntimeContext, In, InNDim, InDims, Out, OutNDim,
                             OutDims, Axis);

  /// Return: void
  return Status;
}

} // namespace Executor
} // namespace SSVM