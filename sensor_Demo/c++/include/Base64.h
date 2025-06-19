#pragma once

#include <string>
#include <vector>

class Base64 {
public:
    static std::string encode(const std::vector<unsigned char>& data);
    static std::vector<unsigned char> decode(const std::string& encoded);
    
    static std::string encodeFile(const std::string& filePath);
};