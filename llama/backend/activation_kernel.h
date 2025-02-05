#pragma once
#include <bmengine/core/core.h>

namespace nn {

using namespace bmengine;

void gate_mul_inplace(const core::Context &ctx,
                      core::Tensor &inp,
                      const core::Tensor &in2,
                      const std::string &gate_type);

} // namespace nn
