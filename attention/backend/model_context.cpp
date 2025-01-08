#include "backend/model_context.h"

namespace model {

using bmengine::core::Context;

ModelContext::ModelContext(Context &&ctx, const ModelBase &m, int batch_size, bool parallel, bool BSHD) :
    Context(std::move(ctx)), model_(m), cfg(m.cfg), parallel_(parallel) {
    set_BSHD(BSHD);
    // TODO ...
}

ModelContext::~ModelContext() {
    if (debug() >= 1) {
        std::cerr << "ModelContext accumulated"
                  << " used_memory " << (used_memory() / 1000) << "KBytes"
                  << " peak_memory " << (peak_memory() / 1000) << "KBytes" << std::endl;
    }
}

} // namespace model
