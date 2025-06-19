#include "Logger.h"
#include <iomanip>
#include <sstream>

Logger::Logger() {}

Logger::~Logger() {
    if (logFile.is_open()) {
        logFile.close();
    }
}

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

void Logger::setLogFile(const std::string& filename) {
    std::lock_guard<std::mutex> lock(logMutex);
    
    if (logFile.is_open()) {
        logFile.close();
    }
    
    logFile.open(filename, std::ios::app);
    if (!logFile.is_open()) {
        std::cerr << "无法打开日志文件: " << filename << std::endl;
    }
}

void Logger::log(LogLevel level, const std::string& message) {
    std::lock_guard<std::mutex> lock(logMutex);
    
    std::string timestamp = getTimestamp();
    std::string levelStr = levelToString(level);
    std::string logMessage = timestamp + " - " + levelStr + " - " + message;
    
    if (logFile.is_open()) {
        logFile << logMessage << std::endl;
    }
    
    if (consoleOutput) {
        std::cout << logMessage << std::endl;
    }
}

void Logger::info(const std::string& message) {
    log(LogLevel::INFO, message);
}

void Logger::warning(const std::string& message) {
    log(LogLevel::WARNING, message);
}

void Logger::error(const std::string& message) {
    log(LogLevel::ERROR, message);
}

std::string Logger::getTimestamp() {
    auto now = std::time(nullptr);
    auto tm = *std::localtime(&now);
    
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

std::string Logger::levelToString(LogLevel level) {
    switch (level) {
        case LogLevel::INFO:
            return "INFO";
        case LogLevel::WARNING:
            return "WARNING";
        case LogLevel::ERROR:
            return "ERROR";
        default:
            return "UNKNOWN";
    }
}