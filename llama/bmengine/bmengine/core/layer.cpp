#include "bmengine/core/layer.h"
#include <iomanip>
#include "bmengine/core/context.h"
#include "bmengine/functions/init.h"
#include <curand.h>
#include <iostream>

namespace bmengine {

namespace core {

void Layer::add_submodule(const std::string &name, Layer *module) {
    module->name = name;
    modules.insert(std::make_pair(name, module));
    module_names.push_back(name);
}

void Layer::add_parameter(const std::string &name, Tensor &t) {
    parameters.insert(std::make_pair(name, &t));
    param_names.push_back(name);
    std::cout << "********************* add parameter: " << name << std::endl;
    std::cout << "param_names.size(): " << param_names.size() << std::endl;
}

void print_layers(std::ostream &os, const Layer *layer, int depth) {
    os << ": (" << layer->layer_type() << ")";

    for (auto &p : layer->parameters) {
        os << std::endl
           << std::setw((depth + 1) * 4) << "";
        os << p.first << " [";
        bool first = true;
        for (auto &d : p.second->size()) {
            if (first)
                first = false;
            else
                os << ", ";
            os << d;
        }
        os << "] dtype=" << get_data_type_name(p.second->dtype())
           << " device=" << p.second->device();
    }
    for (auto &m : layer->modules) {
        os << std::endl
           << std::setw((depth + 1) * 4) << "";
        os << m.first;
        print_layers(os, m.second, depth + 1);
    }
}

std::ostream &operator<<(std::ostream &os, const Layer &layer) {
    print_layers(os, &layer, 0);
    return os;
}

void Layer::init_parameters(const Context &ctx, curandGenerator_t &gen, const std::string &prefix) {
    for (auto &p : parameters) {
        ctx.init_parameter(prefix + "." + p.first, p.second);
        functions::normal_(ctx, gen, *p.second);
    }
    for (auto &m : modules) {
        m.second->init_parameters(ctx, gen, prefix + "." + m.first);
    }
}

std::map<const std::string, Tensor *> Layer::named_parameters(
    const std::string &prefix, bool recursive) {
    std::map<const std::string, Tensor *> named_params;
    std::string layer_prefix = prefix;
    if ((prefix.size() > 0) && (prefix[prefix.size() - 1] != '.')) {
        layer_prefix = prefix + ".";
    }
    for (auto &p_name : param_names) {
        named_params.emplace(layer_prefix + p_name, parameters.at(p_name));
    }
    if (recursive) {
        for (auto &m_name : module_names) {
            for (auto &p : modules[m_name]->named_parameters(m_name + ".", recursive)) {
                named_params.emplace(layer_prefix + p.first, p.second);
            }
        }
    }
    return named_params;
}

void Layer::load_state_dict(
    const Context &ctx,
    const std::map<std::string, const Tensor> &state_dict,
    const std::string &prefix,
    bool allow_missing) {
    std::cout << ">>>> Layer::load_state_dict, prefix: " << prefix << std::endl;
    std::cout << "param_names.size(): " << param_names.size() << std::endl;
    this->prefix = prefix;
    for (auto &p_name : param_names) {
        std::string name = prefix + "." + p_name; // full name
        std::cout << ">>>> Load param: " << name << std::endl;
        if (ctx.debug() >= 2) {
            std::cout << "Load param: " << name << std::endl;
        }
        load_param_from_state_dict(ctx, state_dict, name, parameters[p_name], allow_missing);
    }
    // load recursively
    for (auto &m_name : module_names) {
        std::cout << "Load module: " << m_name << std::endl;
        std::cout << modules[m_name] << std::endl;
        modules[m_name]->load_state_dict(ctx, state_dict, prefix + "." + m_name, allow_missing);
    }
}

void Layer::load_param_from_state_dict(
    const Context &ctx,
    const std::map<std::string, const Tensor> &state_dict,
    const std::string &name,
    Tensor *param,
    bool allow_missing) {
    auto it = state_dict.find(name);
    if (it == state_dict.end()) {
        BM_ASSERT(allow_missing, "param " + name + " not found in state_dict");
        return;
    }
    ctx.assign_or_copy(param, &it->second);
}

}

} // namespace bmengine::core
