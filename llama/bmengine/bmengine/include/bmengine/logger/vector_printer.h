#pragma once

#include <iostream>     // std::cout
#include <iterator>     // std::ostream_iterator
#include <vector>       // std::vector
#include <algorithm>    // std::copy

namespace bmengine {
template <class T>
class PrintWrapper {
 public:
    const std::vector<T> &v;
    PrintWrapper(const std::vector<T> &v): v(v) {}
    ~PrintWrapper() {}
};
}

template <class T>
std::ostream& operator<<(std::ostream& os, const bmengine::PrintWrapper<T>& vw) {
    std::ostream_iterator<int> out_it(os, ", ");
    os << "[";
    std::copy(vw.v.begin(), vw.v.end(), out_it);
    os << "]";
    return os;
}
