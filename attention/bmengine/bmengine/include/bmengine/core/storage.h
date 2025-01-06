#pragma once
#include "bmengine/core/tensor.h"
#include "bmengine/core/export.h"
#include <string>
#include <vector>

namespace bmengine {

namespace core {

enum class DataType;

class BMENGINE_EXPORT ParameterData {
public:
    std::string name;
    std::vector<size_t> shape;
    DataType dtype;
    char* ptr;
    size_t nbytes;
    bool own { true };

    ParameterData(const std::string& name, const std::vector<size_t>& shape, DataType dtype);
    ~ParameterData();
    ParameterData(ParameterData&&);
    ParameterData(const ParameterData&) = delete;
};

class BMENGINE_EXPORT Storage {
public:
    Storage();
    virtual ~Storage();
    Storage(const Storage&) = delete;
    Storage(Storage&&) = delete;
    virtual void fetch_parameter(ParameterData& data);
    virtual void fetch_parameter(const std::string& name, Tensor& data);
    virtual size_t used_memory() const;
};

}

}