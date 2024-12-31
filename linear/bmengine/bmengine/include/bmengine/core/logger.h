#pragma once
#include "bmengine/core/export.h"
#include <string>
#include <sstream>
namespace bmengine {
namespace core {

enum class LogLevel { kLogInfo, kLogWarning, kLogError, kLogDebug, kLogCritical, kLogOff };

class LogLine;
class BMENGINE_EXPORT Logger {
public:
    virtual ~Logger() = default;
    virtual void info(const std::string& message) = 0;
    virtual void warn(const std::string& message) = 0;
    virtual void error(const std::string& message) = 0;
    virtual void debug(const std::string& message) = 0;
    virtual void critical(const std::string& message) = 0;
    virtual void set_log_level(LogLevel level) = 0;

    LogLine info();
    LogLine warn();
    LogLine error();
    LogLine debug();
    LogLine critical();
};

class BMENGINE_EXPORT LoggerFactory {
public:
    virtual ~LoggerFactory() = default;
    virtual Logger* create_logger(const std::string& name) = 0;
    virtual void set_log_level(LogLevel level) = 0;
};

class BMENGINE_EXPORT LogLine {
private:
    Logger* logger;
    int lvl;
    std::stringstream ss;

public:
    LogLine(Logger* logger, int lvl);
    ~LogLine() noexcept;
    LogLine(const LogLine&) = delete;
    LogLine(LogLine&& other);
    template<typename T>
    LogLine& operator<<(const T& t) {
        ss << t;
        return *this;
    }
};

}
}