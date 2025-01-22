#include "backend/model_context.h"
#include "backend/allocate_utils.h"
#include "backend/utils.h"
#include <utility>
#include <numeric>

namespace model {

using namespace bmengine;

ModelContext::ModelContext(core::Context &&ctx,
                           const ModelBase &m,
                           int batch_size,
                           bool parallel,
                           bool BSHD) :
    Context(std::move(ctx)),
    model_(m), cfg(m.cfg), parallel_(parallel) {
    set_BSHD(BSHD);
    layer_devices = model::partition_layer_devices(*this, m.num_layers);
    buf_ctx_ = std::make_shared<TransformerBufferContext>(m, batch_size, parallel, world_size(), BSHD);
    buf_ctx_->set_layer_devices(layer_devices);
    latent_cache_ = cfg.kv_lora_rank > 0 && utils::get_int_env("LATENT_CACHE", 0) == 1;
}

ModelContext::~ModelContext() {
    std::cout << "debug: " << debug() << std::endl;
    if (debug() >= 1) {
        print_memory_summary();
        std::cerr << "ModelContext accumulated"
                  << " used_memory " << (used_memory() / 1000) << "KBytes"
                  << " peak_memory " << (peak_memory() / 1000) << "KBytes" << std::endl;
    }
}

// for batch_generator
ModelContext ModelContext::create(core::Engine &engine,
                                  const ModelBase &md,
                                  const DynBatchConfig &batch_config,
                                  int dev,
                                  bool parallel) {
    std::vector<int> devices(dev == -1 ? engine.num_gpus() : 1);
    std::cout << "ModelContext create " << std::endl;
    if (dev == -1) {
        std::iota(devices.begin(), devices.end(), 0);
    } else {
        devices[0] == dev;
    }
    std::cout << "device: " << devices[0] << std::endl;
}

} // namespace model
