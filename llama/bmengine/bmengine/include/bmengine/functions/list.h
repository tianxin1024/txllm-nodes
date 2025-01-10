#pragma once
#include "bmengine/core/core.h"

namespace bmengine {

namespace functions {

template<typename T>
class ModuleList : public core::Layer {
    BM_LAYER_DEF_PUBLIC(ModuleList)
private:
    std::vector<std::unique_ptr<T>> modules;

public:
    ModuleList() : core::Layer() { }
    ~ModuleList() {
        for (auto it = modules.rbegin(); it != modules.rend(); ++it) {
            it->reset();
        }
    }

    template<typename... Params>
    void append(Params&&... params) {
        modules.emplace_back(std::make_unique<T>(std::forward<Params>(params)...));
        add_submodule(std::to_string(modules.size() - 1), *modules.back());
    }

    T& operator[](size_t i) {
        // size_t is unsigned, don't check >=0 (warning #186-D)
        BM_ASSERT(/*i >= 0 && */ i < modules.size(), "Index out of range");
        return *modules[i];
    }

    size_t size() const { return modules.size(); }
};

}

}
