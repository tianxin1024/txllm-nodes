#include "backend/feedforward.h"

using namespace bmengine;
using bmengine::core::DistLayout;
using bmengine::core::Tensor;

class FeedForward::impl {
public:
    class NormalImpl;

    virtual ~impl() = default;
    virtual Tensor forward(const core::Context &ctx, const Tensor &input, bool quant_back) = 0;

    enum class WeightType {
        kIn,
        kGated,
        kOut,
    };
}; // end of class FeedForward::impl

FeedForward::FeedForward(const core::Context &ctx,
                         model::ModelConfig cfg,
                         model::QuantConfig quant_config,
                         bool parallel) {
}

FeedForward::~FeedForward() = default;

void FeedForward::load_state_dict(const core::Context &ctx,
                                  const std::map<std::string, const core::Tensor> &state_dict,
                                  const std::string &prefix,
                                  bool allow_missing) {
}
