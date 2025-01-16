#pragma once
#include <bmengine/core/core.h>
#include "backend/model.h"

namespace model {

/*
* Generation buffers managment
*/
class BufferContext {
protected:
    const ModelBase &model_;
    bool parallel_;

public:
    BufferContext(const ModelBase &md, bool parallel = false) :
        model_(md), parallel_(parallel) {
    }

    ~BufferContext() = default;
    BufferContext(BufferContext &&) = default;

}; // end of class BufferContext

} // namespace model
