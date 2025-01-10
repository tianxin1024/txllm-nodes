#pragma once
#include <iostream>

class PyModelBase {
protected:
    std::string prefix;
    bool parallel_{false};
    bool loaded_{false};

public:
    PyModelBase() {
    }
    PyModelBase(const std::string &prefix, bool parallel) :
        prefix(prefix), parallel_(parallel) {
    }

    virtual ~PyModelBase() = default;
};
