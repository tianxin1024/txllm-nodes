#pragma once

#include <bmengine/core/core.h>
#include <vector>

namespace utils {

using bmengine::core::DataType;
using bmengine::core::DTypeDeducer;

template <typename T, typename DTD = DTypeDeducer<T>>
class Matrix2D {
    T *data;
    size_t len;
    size_t dim0;
    size_t dim1;
    T def_val;

public:
    Matrix2D(size_t dim0, size_t dim1, T def_val = 0) :
        dim0(dim0), dim1(dim1), len(dim0 * dim1), def_val(def_val) {
        data = new T[len];
        std::fill_n(data, len, def_val);
    }
}; // end of class Matrix2D

template class Matrix2D<int32_t>;

} // namespace utils
