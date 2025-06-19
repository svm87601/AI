#include "SensorDataCollector.h"
#include "Logger.h"
#include <iostream>
#include <stdexcept>

int main(int argc, char* argv[]) {
    try {
        SensorDataCollector collector;
        collector.run();
        return 0;
    } catch (const std::exception& e) {
        LOGGER.error("程序启动失败: " + std::string(e.what()));
        std::cerr << "程序启动失败: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        LOGGER.error("程序启动失败: 未知错误");
        std::cerr << "程序启动失败: 未知错误" << std::endl;
        return 1;
    }
}