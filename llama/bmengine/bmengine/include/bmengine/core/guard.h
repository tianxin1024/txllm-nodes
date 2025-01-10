#pragma once
#include "bmengine/core/export.h"

namespace bmengine {

namespace core {

class Context;

class BMENGINE_EXPORT WithDevice {
private:
    const Context* ctx;

public:
    WithDevice(const Context& ctx, int dev);
    ~WithDevice();
    WithDevice(const WithDevice&) = delete;
    WithDevice& operator=(const WithDevice&) = delete;
    WithDevice(WithDevice&&);
    WithDevice& operator=(WithDevice&&) = delete;
};

class BMENGINE_EXPORT ScopeDevice {
private:
    Context* ctx;

public:
    ScopeDevice(const Context& ctx, int dev);
    ~ScopeDevice();
    ScopeDevice(const ScopeDevice&) = delete;
    ScopeDevice& operator=(const ScopeDevice&) = delete;
    ScopeDevice(ScopeDevice&&);
    ScopeDevice& operator=(ScopeDevice&&) = delete;
};

class BMENGINE_EXPORT WithDebug {
private:
    const Context* ctx;
    const int previous_level;

public:
    WithDebug(const Context& ctx, int debug_level);
    ~WithDebug();
    WithDebug(const WithDebug&) = delete;
    WithDebug& operator=(const WithDebug&) = delete;
    WithDebug(WithDebug&&);
    WithDebug& operator=(WithDebug&&) = delete;
};
}

}
