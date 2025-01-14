#include "backend/attention.h"
#include <bmengine/core/core.h>

namespace nn {

using namespace bmengine;

class Attention::impl {
public:
    class NormalImpl;
    impl() = default;
    virtual ~impl() = default;
    impl(const impl &) = delete;
    impl(impl &&) = delete;
}; // end of class Attention::impl

Attention::Attention(const core::Context &ctx,
                     model::ModelConfig cfg,
                     model::QuantConfig quant_config,
                     bool parallel) :
    core::Layer() {
    impl::NormalImpl *normal_impl = nullptr;
}

Attention::~Attention() = default;

void Attention::load_state_dict(const core::Context &ctx,
                                const std::map<std::string, const core::Tensor> &state_dict,
                                const std::string &prefix,
                                bool allow_missing) {
    std::cout << "Attention::load_state_dict" << std::endl;
}

} // namespace nn
