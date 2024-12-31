#pragma once
#include "bmengine/core/core.h"

namespace bmengine {
namespace logger {

class BMENGINE_EXPORT NullLoggerFactory : public core::LoggerFactory {
private:
    class impl;
    std::unique_ptr<impl> pimpl;

public:
    NullLoggerFactory();
    ~NullLoggerFactory();
    core::Logger* create_logger(const std::string& name) override;
    void set_log_level(core::LogLevel level) override;
};

}
}