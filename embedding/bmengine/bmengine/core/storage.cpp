#include "bmengine/core/storage.h"
#include "bmengine/core/tensor.h"
#include "bmengine/core/exception.h"
#include <iostream>

namespace bmengine {

namespace core {

ParameterData::ParameterData(
    const std::string& name, const std::vector<size_t>& shape, DataType dtype)
    : name(name), shape(shape), dtype(dtype) {

    nbytes = get_numel(shape) * get_elem_size(dtype);
    ptr = nullptr;
}
ParameterData::~ParameterData() {
    if (own && ptr != nullptr)
        delete[] ptr;
}
ParameterData::ParameterData(ParameterData&& other) {
    name = other.name;
    shape = other.shape;
    dtype = other.dtype;
    nbytes = other.nbytes;
    ptr = other.ptr;
    own = other.own;
    other.ptr = nullptr;
}

Storage::Storage() = default;
Storage::~Storage() = default;

void Storage::fetch_parameter(ParameterData& data) {
    std::cerr << "Loading " << data.name << std::endl;
    BM_EXCEPTION("fetch_parameter not implemented");
}

void Storage::fetch_parameter(const std::string& name, core::Tensor& data) {
    BM_EXCEPTION("fetch_parameter not implemented");
}

size_t Storage::used_memory() const {
    return 0;
}

} // namespace core

} // namespace bmengine
