#pragma once

#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace bmengine {
namespace logger {

template<class Iter>
inline std::ostream& print_range(std::ostream& out, Iter begin, Iter end) {
    out << "[";
    for (int i = 0; begin != end && i < 80; ++i, ++begin) {
        if (i > 0)
            out << ' ';
        out << *begin;
    }
    if (begin != end) {
        out << " ...";
    }
    return out << "]";
}

template<class T>
inline std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
    return print_range(out, v.begin(), v.end());
}

template<class T>
std::string to_string(const std::vector<T>& v) {
    std::ostringstream os;
    os << v;
    return os.str();
}
using std::to_string;
static inline std::string to_string(const char* str) {
    return std::string(str);
}

static inline std::string to_string(const std::string& str) {
    return str;
}

template<class T1, class T2>
std::string str_cat(const T1& a, const T2& b) {
    return to_string(a) + to_string(b);
}

template<class T1, class T2, class T3>
std::string str_cat(const T1& a, const T2& b, const T3& c) {
    return to_string(a) + to_string(b) + to_string(c);
}

template<class T1, class T2, class T3, class T4>
std::string str_cat(const T1& a, const T2& b, const T3& c, const T4& d) {
    return to_string(a) + to_string(b) + to_string(c) + to_string(d);
}

template<class T1, class T2, class T3, class T4, class T5>
std::string str_cat(const T1& a, const T2& b, const T3& c, const T4& d, const T5& e) {
    return to_string(a) + to_string(b) + to_string(c) + to_string(d) + to_string(e);
}

template<class T1, class T2, class T3, class T4, class T5, class T6>
std::string str_cat(const T1& a, const T2& b, const T3& c, const T4& d, const T5& e, const T6& f) {
    return to_string(a) + to_string(b) + to_string(c) + to_string(d) + to_string(e) + to_string(f);
}

template<class T1, class T2, class T3, class T4, class T5, class T6, class T7>
std::string str_cat(const T1& a, const T2& b, const T3& c, const T4& d, const T5& e, const T6& f, const T7& g) {
    return to_string(a) + to_string(b) + to_string(c) + to_string(d) + to_string(e) + to_string(f) + to_string(g);
}


} // namespace logger
} // namespace bmengine

namespace std {
using bmengine::logger::operator<<;
using bmengine::logger::to_string;
}

using std::endl;