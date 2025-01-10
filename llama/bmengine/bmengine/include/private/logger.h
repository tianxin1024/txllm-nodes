#include "bmengine/core/core.h"
#include "spdlog/spdlog.h"
namespace bmengine {
namespace logger {

class StandardLogger : public core::Logger {
    std::shared_ptr<spdlog::logger> logger;

public:
    StandardLogger(std::shared_ptr<spdlog::logger> logger) : logger(logger) {
        logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%n] %v");
    }
    ~StandardLogger() = default;

    void set_log_level(core::LogLevel level) override {
        switch (level) {
            case core::LogLevel::kLogDebug: logger->set_level(spdlog::level::debug); break;
            case core::LogLevel::kLogInfo: logger->set_level(spdlog::level::info); break;
            case core::LogLevel::kLogWarning: logger->set_level(spdlog::level::warn); break;
            case core::LogLevel::kLogError: logger->set_level(spdlog::level::err); break;
            case core::LogLevel::kLogCritical: logger->set_level(spdlog::level::critical); break;
            default: logger->set_level(spdlog::level::off); break;
        }
    }

    void info(const std::string& message) noexcept override { logger->info(message); }
    void warn(const std::string& message) noexcept override { logger->warn(message); }
    void error(const std::string& message) noexcept override { logger->error(message); }
    void debug(const std::string& message) noexcept override { logger->debug(message); }
    void critical(const std::string& message) noexcept override { logger->critical(message); }
};

}
}
