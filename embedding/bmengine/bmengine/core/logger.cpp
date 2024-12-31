#include "bmengine/core/logger.h"
#include <iostream>

namespace bmengine {
namespace core {

LogLine::LogLine(Logger* logger, int lvl) : logger(logger), lvl(lvl), ss() { }
LogLine::~LogLine() {
    try {
        if (logger) {
            if (lvl == 0)
                logger->info(ss.str());
            else if (lvl == 1)
                logger->warn(ss.str());
            else if (lvl == 2)
                logger->error(ss.str());
            else if (lvl == 3)
                logger->debug(ss.str());
            else if (lvl == 4)
                logger->critical(ss.str());
        }
    } catch (const std::exception& err) { std::cerr << err.what() << std::endl; }
}
LogLine::LogLine(LogLine&& other) : logger(other.logger), lvl(other.lvl), ss(std::move(other.ss)) {
    other.logger = nullptr;
}

LogLine Logger::info() {
    return LogLine(this, 0);
}
LogLine Logger::warn() {
    return LogLine(this, 1);
}
LogLine Logger::error() {
    return LogLine(this, 2);
}
LogLine Logger::debug() {
    return LogLine(this, 3);
}
LogLine Logger::critical() {
    return LogLine(this, 4);
}

}

}