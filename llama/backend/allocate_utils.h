#pragma once

#include <vector>
#include <bmengine/core/core.h>

namespace model {

/**
 * Partition layers equally to all devices.
 * return device of each layer.
 */
inline std::vector<int> partition_layer_devices(
    const bmengine::core::Context &ctx, int num_layers) {
    int num_dev = (int)ctx.devices().size();
    int layers_per_dev = (num_layers + num_dev - 1) / num_dev;

    std::vector<int> devices(num_layers);
    for (int i = 0; i < num_layers; i++) {
        devices[i] = i / layers_per_dev;
    }
    return devices;
}

} // namespace model
