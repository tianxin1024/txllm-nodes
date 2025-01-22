#pragma once
#include <iostream>
#include "backend/model.h"

class PyModelBase {
protected:
    std::string prefix;
    bool parallel_{false};
    bool loaded_{false};

    virtual void on_load() {
        loaded_ = true;
        engine()->freeze_model_memory();
    }

public:
    PyModelBase() {
    }
    PyModelBase(const std::string &prefix, bool parallel) :
        prefix(prefix), parallel_(parallel) {
    }

    virtual ~PyModelBase() = default;

    virtual bmengine::core::Engine *engine() = 0;

    virtual std::vector<model::ModelBase *> par_models() = 0;

    bool is_parallel() const {
        return parallel_;
    }
};
