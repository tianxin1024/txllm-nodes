#pragma once
#include "bmengine/core/core.h"

namespace bmengine {
namespace logger {

class BMENGINE_EXPORT SyslogLoggerFactory : public core::LoggerFactory {
private:
    class impl;
    std::unique_ptr<impl> pimpl;

public:
    SyslogLoggerFactory(const std::string& ident);
    ~SyslogLoggerFactory();
    core::Logger* create_logger(const std::string& name) override;
    void set_log_level(core::LogLevel level) override;
};

}
}