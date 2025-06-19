#pragma once

#include <string>
#include <memory>
#include <nlohmann/json.hpp>
#include <mosquitto.h>
#include <libcamera/libcamera.h>

using json = nlohmann::json;

class SensorDataCollector {
public:
    SensorDataCollector(const std::string& configFile = "config.json");
    ~SensorDataCollector();
    
    void run();
    
private:
    // 配置相关
    json config;
    json loadConfig(const std::string& configFile);
    
    // 连接相关
    int serialFd;
    struct mosquitto* mqttClient;
    std::unique_ptr<libcamera::CameraManager> cameraManager;
    std::shared_ptr<libcamera::Camera> camera;
    
    void setupConnections();
    void setupCamera();
    
    // 数据采集相关
    std::string readSensorData(int timeout = 2);
    std::string captureImage();
    bool saveToJsFile(const json& data, const std::string& filename = "sensorData.js");
    bool publishMqtt(const json& data);
    void collectAndSendData();
    
    // 清理资源
    void cleanup();
};