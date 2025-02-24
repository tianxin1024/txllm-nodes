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

    size_t size() const {
        return len;
    }
    size_t dim(size_t i) const {
        return i == 0 ? dim0 : dim1;
    }

    T &operator()(size_t x, size_t y) {
        return data[x * dim1 + y];
    }

    bmengine::core::Tensor to_tensor(const bmengine::core::Context &ctx) {
        auto tensor = ctx.tensor({dim0, dim1}, DTD::data_type());
        tensor.from_buffer(data);
        return tensor;
    }

}; // end of class Matrix2D

template class Matrix2D<int32_t>;

template <typename T, typename DTT = DTypeDeducer<T>>
class Matrix3D {
    T *data;
    size_t len;
    size_t dim0;
    size_t dim1;
    size_t dim2;
    size_t stride0;

public:
    Matrix3D(size_t dim0, size_t dim1, size_t dim2, T default_val = 0) :
        dim0(dim0), dim1(dim1), dim2(dim2), len(dim0 * dim1 * dim2) {
        data = new T[len];
        stride0 = dim1 * dim2;
        std::fill_n(data, len, default_val);
    }

    void resize(size_t new_dim0, size_t new_dim1, size_t new_dim2) {
        dim0 = new_dim0, dim1 = new_dim1, dim2 = new_dim2;
        if (dim0 * dim1 * dim2 > len) {
            delete[] data;
            data = new T[dim0 * dim1 * dim2];
        }
        len = dim0 * dim1 * dim2;
        stride0 = dim1 * dim2;
        std::fill_n(data, len, 0);
    }

    T &operator()(size_t x, size_t y, size_t z) {
        return data[x * stride0 + y * dim2 + z];
    }

    bmengine::core::Tensor to_tensor(bmengine::core::Context &ctx) {
        auto tensor = ctx.tensor({dim0, dim1, dim2}, DTT::data_type());
        tensor.from_buffer(data);
        return tensor;
    }
};

} // namespace utils
