cmake_minimum_required(VERSION 3.10)
project(SensorDataCollector VERSION 1.0)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖包
find_package(PkgConfig REQUIRED)
pkg_check_modules(MOSQUITTO REQUIRED libmosquitto)
pkg_check_modules(LIBCAMERA REQUIRED libcamera)

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${MOSQUITTO_INCLUDE_DIRS})
include_directories(${LIBCAMERA_INCLUDE_DIRS})

# 添加源文件
set(SOURCES
    src/main.cpp
    src/SensorDataCollector.cpp
    src/Base64.cpp
    src/Logger.cpp
)

# 创建可执行文件
add_executable(sensor_collector ${SOURCES})

# 链接库
target_link_libraries(sensor_collector
    ${MOSQUITTO_LIBRARIES}
    ${LIBCAMERA_LIBRARIES}
    pthread
)

# 安装目标
install(TARGETS sensor_collector DESTINATION bin)
install(FILES config.json DESTINATION etc/sensor_collector)