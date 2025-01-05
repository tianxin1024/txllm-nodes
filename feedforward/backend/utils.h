#pragma once
#include <bmengine/core/core.h>

namespace model {

namespace core = bmengine::core;

core::Tensor convert_fp32(const core::Context &ctx, const core::Tensor &logits);

} // namespace model

namespace utils {

static inline int get_int_env(const char *name, int def_val = 0) {
    char *env_str = std::getenv(name);
    return env_str != nullptr ? std::atoi(env_str) : def_val;
}

} // namespace utils
