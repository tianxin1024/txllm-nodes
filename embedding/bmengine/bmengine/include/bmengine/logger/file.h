#pragma once
#include "bmengine/core/core.h"

namespace bmengine {
namespace logger {

class BMENGINE_EXPORT RotateFileLoggerFactory : public core::LoggerFactory {
private:
    class impl;
    std::unique_ptr<impl> pimpl;

public:
    RotateFileLoggerFactory(const std::string& filename, size_t rotate_size, int max_files);
    ~RotateFileLoggerFactory();
    core::Logger* create_logger(const std::string& name) override;
    void set_log_level(core::LogLevel level) override;
};

class BMENGINE_EXPORT DailyFileLoggerFactory : public core::LoggerFactory {
private:
    class impl;
    std::unique_ptr<impl> pimpl;

public:
    DailyFileLoggerFactory(const std::string& filename, int hour, int minute);
    ~DailyFileLoggerFactory();
    core::Logger* create_logger(const std::string& name) override;
    void set_log_level(core::LogLevel level) override;
};

}
}