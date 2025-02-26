#include "backend/model_context.h"
#include "backend/allocate_utils.h"
#include "backend/dyn_batch_context.h"
#include "backend/rag_buffer_context.h"
#include <bmengine/functions/all.h>
#include "backend/utils.h"
#include <utility>
#include <numeric>

namespace model {

using namespace bmengine;
using bmengine::core::Context;

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

void ModelContext::set_host_reducer(std::shared_ptr<HostAllReducer> reducer) {
    // host_reducer_ = std::move(reducer);
    cudaStream_t red_stream;
    // TODO ...
}

KVCacheConfig ModelContext::get_kv_cache_config() {
    int num_kv_heads = parallel_ ? cfg.num_kv_heads / world_size() : cfg.num_kv_heads;
    int dim_head = cfg.dim_head;
    core::DataType dtype = cfg.dtype;
    std::shared_ptr<core::DataType> scale_dtype;
    char *env = std::getenv("KV_CACHE_DTYPE");
    if (env && *env) {
        if (std::string("int8") == env) {
            dtype = core::DataType::kInt8;
            scale_dtype = std::make_shared<core::DataType>(core::DataType::kFloat);
        } else {
            throw std::runtime_error("Unsupported dtype: " + std::string(env));
        }
    }
    is_BSHD();
    KVCacheConfig cache_config =
        {cfg.num_layers, num_kv_heads, dim_head, dtype, is_BSHD(), scale_dtype};
    cache_config.layer_devices = layer_devices;
    return cache_config;
}

void ModelContext::resize_task_buf(int b, size_t new_length) {
    rag_buffer()->resize_task_buf(*this, b, new_length);
}

void ModelContext::free_task_buf(int b) {
    rag_buffer()->free_task_buf(b);
}

// for batch_generator
ModelContext ModelContext::create(core::Engine &engine,
                                  const ModelBase &md,
                                  const DynBatchConfig &batch_config,
                                  int dev,
                                  bool parallel) {
    std::cout << ">>>>>>>>>> dev: " << dev << std::endl;
    std::vector<int> devices(dev == -1 ? engine.num_gpus() : 1);
    std::cout << "ModelContext create " << std::endl;
    if (dev == -1) {
        std::iota(devices.begin(), devices.end(), 0);
    } else {
        devices[0] = dev;
    }
    std::cout << "device: " << devices[0] << std::endl;

    core::Context ctx = parallel ? engine.create_context_rank(dev) : engine.create_context(devices);
    ModelContext model_ctx(std::move(ctx), md, batch_config.max_batch, parallel);
    model_ctx.set_BSHD(batch_config.flash_attention);
    model_ctx.dyn_batch_ = std::make_shared<DynBatchContext>();
    if (batch_config.rag_buffer) {
        auto k_cfg = model_ctx.get_kv_cache_config();
        auto v_cfg = k_cfg;
        if (model_ctx.latent_cache_ && model_ctx.cfg.kv_lora_rank > 0) {
            k_cfg.num_heads = 1;
            v_cfg.num_heads = 1;
            k_cfg.dim_head = model_ctx.cfg.kv_lora_rank + model_ctx.cfg.qk_rope_head_dim;
            v_cfg.dim_head = 0;
        }
        model_ctx.set_rag_buffer(std::make_shared<RagBufferContext>(k_cfg, v_cfg));
    }
    model_ctx.reducer_ = std::make_shared<ReduceContext>();
    return model_ctx;
}

void ModelContext::update_act_scale(const std::string &name, const Tensor &act) {
    Tensor x = functions::reduce_abs_max(*this, act);
    if (act_scale_map_.count(name) > 0) {
        x = functions::BinaryElementwiseOp(*this, functions::BinaryElementwiseOp::Max).forward(*this, act_scale_map_[name], x);
    }
    act_scale_map_[name] = x;
}

} // namespace model
