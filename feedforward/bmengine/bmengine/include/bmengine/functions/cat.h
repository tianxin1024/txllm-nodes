#pragma once
#include "bmengine/core/core.h"

namespace bmengine {

namespace functions {

void cat(
    const core::Context& ctx,
    const core::Tensor& x,
    const core::Tensor& y,
    const core::Tensor& out,
    int dim);

}
}