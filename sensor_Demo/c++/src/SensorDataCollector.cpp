#include "SensorDataCollector.h"
#include "Logger.h"
#include "Base64.h"

#include <fstream>
#include <chrono>
#include <thread>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <libcamera/camera_manager.h>
#include <libcamera/camera.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/control_ids.h>
#include <libcamera/controls.h>
#include <libcamera/formats.h>

SensorDataCollector::SensorDataCollector(const std::string& configFile)
    : serialFd(-1), mqttClient(nullptr) {
    // 加载配置
    config = loadConfig(configFile);
    
    // 设置日志
    LOGGER.setLogFile("sensor_data.log");
    
    // 建立连接
    setupConnections();
    setupCamera();
}

SensorDataCollector::~SensorDataCollector() {
    cleanup();
}

json SensorDataCollector::loadConfig(const std::string& configFile) {
    try {
        std::ifstream f(configFile);
        if (!f.is_open()) {
            LOGGER.error("配置文件 " + configFile + " 不存在");
            throw std::runtime_error("配置文件不存在");
        }
        return json::parse(f);
    } catch (const json::parse_error& e) {
        LOGGER.error("配置文件格式错误: " + std::string(e.what()));
        throw;
    }
}

void SensorDataCollector::setupConnections() {
    // 初始化串口
    try {
        auto serialConfig = config["serial"];
        std::string port = serialConfig["port"];
        int baudrate = serialConfig["baudrate"];
        int timeout = serialConfig["timeout"];
        
        serialFd = open(port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
        if (serialFd == -1) {
            LOGGER.error("无法打开串口: " + port);
            throw std::runtime_error("串口打开失败");
        }
        
        struct termios options;
        tcgetattr(serialFd, &options);
        
        // 设置波特率
        cfsetispeed(&options, B115200);  // 根据实际波特率设置
        cfsetospeed(&options, B115200);
        
        // 设置控制模式
        options.c_cflag |= (CLOCAL | CREAD);
        options.c_cflag &= ~PARENB;  // 无奇偶校验
        options.c_cflag &= ~CSTOPB;  // 1个停止位
        options.c_cflag &= ~CSIZE;
        options.c_cflag |= CS8;       // 8位数据位
        
        // 设置本地模式和输入模式
        options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
        options.c_iflag &= ~(IXON | IXOFF | IXANY);
        
        // 设置输出模式
        options.c_oflag &= ~OPOST;
        
        // 设置超时
        options.c_cc[VMIN] = 0;
        options.c_cc[VTIME] = timeout * 10;  // 超时时间，单位是十分之一秒
        
        // 应用设置
        tcsetattr(serialFd, TCSANOW, &options);
        
        LOGGER.info("串口连接成功");
    } catch (const std::exception& e) {
        LOGGER.error("串口连接失败: " + std::string(e.what()));
        throw;
    }
    
    // 初始化MQTT客户端
    try {
        auto mqttConfig = config["mqtt"];
        std::string broker = mqttConfig["broker"];
        int port = mqttConfig["port"];
        std::string username = mqttConfig["username"];
        std::string password = mqttConfig["password"];
        
        mosquitto_lib_init();
        mqttClient = mosquitto_new(nullptr, true, nullptr);
        if (!mqttClient) {
            LOGGER.error("MQTT客户端创建失败");
            throw std::runtime_error("MQTT客户端创建失败");
        }
        
        mosquitto_username_pw_set(mqttClient, username.c_str(), password.c_str());
        
        int rc = mosquitto_connect(mqttClient, broker.c_str(), port, 60);
        if (rc != MOSQ_ERR_SUCCESS) {
            LOGGER.error("MQTT连接失败: " + std::string(mosquitto_strerror(rc)));
            throw std::runtime_error("MQTT连接失败");
        }
        
        LOGGER.info("MQTT连接成功");
    } catch (const std::exception& e) {
        LOGGER.error("MQTT连接失败: " + std::string(e.what()));
        throw;
    }
}

void SensorDataCollector::setupCamera() {
    try {
        auto cameraSetting = config["camera"];  // 修改这个变量名
        int width = cameraSetting["width"];
        int height = cameraSetting["height"];
        
        cameraManager = std::make_unique<libcamera::CameraManager>();
        cameraManager->start();
        
        if (cameraManager->cameras().empty()) {
            LOGGER.error("没有找到可用的摄像头");
            return;
        }
        
        camera = cameraManager->cameras()[0];
        camera->acquire();
        
        std::unique_ptr<libcamera::CameraConfiguration> cameraConfig = 
            camera->generateConfiguration({libcamera::StreamRole::StillCapture});
        libcamera::StreamConfiguration& streamConfig = cameraConfig->at(0);
        streamConfig.size.width = width;
        streamConfig.size.height = height;
        streamConfig.pixelFormat = libcamera::formats::RGB888;
        
        camera->configure(cameraConfig.get());
        camera->start();
        
        LOGGER.info("摄像头初始化成功");
    } catch (const std::exception& e) {
        LOGGER.error("摄像头初始化失败: " + std::string(e.what()));
        camera = nullptr;
    }
}

std::string SensorDataCollector::readSensorData(int timeout) {
    if (serialFd < 0) {
        return "";
    }
    
    try {
        // 清空输入缓冲区
        tcflush(serialFd, TCIFLUSH);
        
        char buffer[1024] = {0};
        std::string data;
        auto startTime = std::chrono::steady_clock::now();
        
        while (true) {
            int bytesAvailable;
            ioctl(serialFd, FIONREAD, &bytesAvailable);
            
            if (bytesAvailable > 0) {
                int bytesRead = read(serialFd, buffer, sizeof(buffer) - 1);
                if (bytesRead > 0) {
                    buffer[bytesRead] = '\0';
                    data += buffer;
                    
                    if (data.find("\r\n") != std::string::npos) {
                        break;
                    }
                }
            }
            
            auto currentTime = std::chrono::steady_clock::now();
            auto elapsedTime = std::chrono::duration_cast<std::chrono::seconds>
                (currentTime - startTime).count();
            
            if (elapsedTime > timeout) {
                LOGGER.warning("读取传感器数据超时");
                break;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        // 处理数据
        std::string result;
        if (!data.empty()) {
            size_t pos = data.find("\r\n");
            if (pos != std::string::npos) {
                result = data.substr(0, pos);
            } else {
                result = data;
            }
        }
        
        LOGGER.info("传感器数据: " + result);
        return result;
    } catch (const std::exception& e) {
        LOGGER.error("读取传感器数据失败: " + std::string(e.what()));
        return "";
    }
}

std::string SensorDataCollector::captureImage() {
    if (!camera) {
        LOGGER.error("摄像头未初始化，无法拍照");
        return "";
    }
    
    try {
        // 在使用系统调用前释放摄像头资源
        camera->stop();
        camera->release();
        
        // 获取配置的分辨率
        auto cameraSetting = config["camera"];
        int width = cameraSetting["width"];
        int height = cameraSetting["height"];
        
        // 使用系统调用方式拍照，添加分辨率参数
        std::string imagePath = "capture.jpg";
        std::string command = "libcamera-still -o " + imagePath + " --width " + std::to_string(width) + 
                             " --height " + std::to_string(height) + " --immediate -n";
        system(command.c_str());
        
        // 读取并编码图片
        std::string encodedImage = Base64::encodeFile(imagePath);
        
        // 重新获取摄像头资源
        camera->acquire();
        
        // 重新配置摄像头
        std::unique_ptr<libcamera::CameraConfiguration> cameraConfig = 
            camera->generateConfiguration({libcamera::StreamRole::StillCapture});
        libcamera::StreamConfiguration& streamConfig = cameraConfig->at(0);
        streamConfig.size.width = width;
        streamConfig.size.height = height;
        streamConfig.pixelFormat = libcamera::formats::RGB888;
        
        camera->configure(cameraConfig.get());
        camera->start();
        
        LOGGER.info("图片拍摄成功");
        return encodedImage;
    } catch (const std::exception& e) {
        LOGGER.error("拍照失败: " + std::string(e.what()));
        return "";
    }
}

bool SensorDataCollector::saveToJsFile(const json& data, const std::string& filename) {
    try {
        std::ofstream file(filename);
        if (!file.is_open()) {
            LOGGER.error("无法打开文件: " + filename);
            return false;
        }
        
        file << "var sensorData = " << data.dump(2) << ";";
        file.close();
        
        LOGGER.info("数据已保存到 " + filename);
        return true;
    } catch (const std::exception& e) {
        LOGGER.error("保存文件失败: " + std::string(e.what()));
        return false;
    }
}

bool SensorDataCollector::publishMqtt(const json& data) {
    if (!mqttClient) {
        return false;
    }
    
    try {
        std::string topic = config["mqtt"]["topic"];
        std::string payload = data.dump();
        
        int rc = mosquitto_publish(mqttClient, nullptr, topic.c_str(), 
                                  payload.size(), payload.c_str(), 0, false);
        
        if (rc != MOSQ_ERR_SUCCESS) {
            LOGGER.error("MQTT发布失败: " + std::string(mosquitto_strerror(rc)));
            return false;
        }
        
        LOGGER.info("MQTT数据发布成功");
        return true;
    } catch (const std::exception& e) {
        LOGGER.error("MQTT发布失败: " + std::string(e.what()));
        return false;
    }
}

void SensorDataCollector::collectAndSendData() {
    LOGGER.info("开始数据采集");
    
    // 读取传感器数据
    std::string soilData = readSensorData();
    
    // 拍摄照片
    std::string imageBase64 = captureImage();
    
    // 组合数据
    json data = {
        {"timestamp", std::chrono::system_clock::now().time_since_epoch().count() / 1000000},
        {"soil_data", soilData},
        {"image_base64", imageBase64}
    };
    
    // 保存到文件
    saveToJsFile(data);
    
    // 发布到MQTT
    publishMqtt(data);
    
    LOGGER.info("数据采集完成");
}

void SensorDataCollector::run() {
    LOGGER.info("传感器数据采集器启动");
    
    try {
        while (true) {
            collectAndSendData();
            
            // 等待指定间隔
            int interval = config.value("interval", 30);
            LOGGER.info("等待 " + std::to_string(interval) + " 秒后进行下次采集");
            std::this_thread::sleep_for(std::chrono::seconds(interval));
        }
    } catch (const std::exception& e) {
        LOGGER.error("运行时错误: " + std::string(e.what()));
    } catch (...) {
        LOGGER.error("未知错误");
    }
    
    cleanup();
}

void SensorDataCollector::cleanup() {
    // 清理摄像头资源
    if (camera) {
        try {
            camera->stop();
            camera->release();
            camera.reset();
            LOGGER.info("摄像头已关闭");
        } catch (const std::exception& e) {
            LOGGER.error("关闭摄像头时出错: " + std::string(e.what()));
        }
    }
    
    if (cameraManager) {
        cameraManager->stop();
    }
    
    // 清理串口资源
    if (serialFd >= 0) {
        close(serialFd);
        serialFd = -1;
        LOGGER.info("串口已关闭");
    }
    
    // 清理MQTT资源
    if (mqttClient) {
        mosquitto_disconnect(mqttClient);
        mosquitto_destroy(mqttClient);
        mosquitto_lib_cleanup();
        mqttClient = nullptr;
        LOGGER.info("MQTT连接已断开");
    }
}