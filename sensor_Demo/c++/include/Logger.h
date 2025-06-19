#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <mutex>
#include <ctime>

enum class LogLevel {
    INFO,
    WARNING,
    ERROR
};

class Logger {
public:
    static Logger& getInstance();
    
    void setLogFile(const std::string& filename);
    void log(LogLevel level, const std::string& message);
    void info(const std::string& message);
    void warning(const std::string& message);
    void error(const std::string& message);

private:
    Logger();
    ~Logger();
    
    std::string getTimestamp();
    std::string levelToString(LogLevel level);
    
    std::ofstream logFile;
    std::mutex logMutex;
    bool consoleOutput = true;
};

#define LOGGER Logger::getInstance()