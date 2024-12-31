#pragma once
#include "bmengine/core/core.h"

namespace bmengine {
namespace logger {

class BMENGINE_EXPORT StandardLoggerFactory : public core::LoggerFactory {
private:
    class impl;
    std::unique_ptr<impl> pimpl;

public:
    StandardLoggerFactory(bool use_stderr = false);
    ~StandardLoggerFactory();
    core::Logger* create_logger(const std::string& name) override;
    void set_log_level(core::LogLevel level) override;
};

}
}