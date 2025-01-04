#include "bmengine/core/context.h"
#include "bmengine/core/guard.h"
#include <iostream>
#include "private/context.h"

namespace bmengine {

namespace core {

WithDevice::WithDevice(const Context& context, int dev) : ctx(&context) {
    ctx->alloc_device(dev);
}
WithDevice::~WithDevice() {
    if (ctx != nullptr)
        ctx->release_device();
}

WithDevice::WithDevice(WithDevice&& other) : ctx(other.ctx) {
    other.ctx = nullptr;
}

ScopeDevice::ScopeDevice(const Context& context, int dev) : ctx(const_cast<Context*>(&context)) {
    ctx->push_device(dev);
}
ScopeDevice::~ScopeDevice() {
    if (ctx != nullptr)
        ctx->pop_device();
}

ScopeDevice::ScopeDevice(ScopeDevice&& other) : ctx(other.ctx) {
    other.ctx = nullptr;
}

WithDebug::WithDebug(const Context& context, int debug_level)
    : ctx(&context), previous_level(ctx->debug()) {
    ctx->enable_debug(debug_level);
}

WithDebug::~WithDebug() {
    if (ctx != nullptr)
        ctx->enable_debug(previous_level);
}

WithDebug::WithDebug(WithDebug&& other) : ctx(other.ctx), previous_level(other.previous_level) {
    other.ctx = nullptr;
}

} // namespace core

} // namespace bmengine
