#include "backend/utils.h"

namespace utils {

using bmengine::core::DataType;

const std::map<const std::string, DataType> datatype_name_mapping{
    {"double", DataType::kDouble},
    {"float", DataType::kFloat},
    {"half", DataType::kHalf},
    {"bfloat", DataType::kBFloat16},
    {"int8", DataType::kInt8},
    {"int16", DataType::kInt16},
    {"int32", DataType::kInt32},
};

DataType name_to_data_type(const std::string &name) {
    if (datatype_name_mapping.count(name)) {
        return datatype_name_mapping.at(name);
    }
    BM_EXCEPTION("unknown datatype name: " + std::string(name));
}

} // namespace utils
