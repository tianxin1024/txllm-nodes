#include "backend/model_context.h"
#include "backend/utils.h"

namespace model {

class HostAllReducerImpl : public HostAllReducer {
public:
    HostAllReducerImpl() = delete;
    HostAllReducerImpl(int world_size, int num_threads, cudaStream_t stream, size_t buffer_size = 10000 * 8192) {
    }

}; // end of class HostAllReducerImpl

HostAllReducer *ModelContext::create_host_reducer() {
    int num_thread = utils::get_int_env("HOST_REDUCE_THREAD", world_size() == 2 ? 16 : 32);
    auto stream = current_stream()->ptr;
    return new HostAllReducerImpl(world_size(), num_thread, stream);
}

} // namespace model
