#pragma once

#include "backend/attention.h"

namespace nn {

class Attention::impl {
public:
    class NormalImpl;
    impl() = default;
    virtual ~impl() = default;
    impl(const impl &) = delete;
    impl(impl &&) = delete;

    // virtual core::Tensor dynamic_batch_forward(model::ModelContext &ctx,
    //                                            const core::Tensor &hidden_q,
    //                                            const core::Tensor &position_or_bias,
    //                                            core::Tensor *output) {
    //     throw std::runtime_error("Unsupported");
    // }

    virtual void on_load(const core::Context &ctx) {
    }
}; // end of class Attention::impl

} // namespace nn
