#include "config_loader.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <shlobj.h>

ConfigLoader::ConfigLoader(const std::string& filename) : filename_(filename) {}

bool ConfigLoader::load() {
    std::ifstream file(filename_);
    if (!file.is_open()) {
        return false;
    }
    
    std::string current_section;
    std::string line;
    
    while (std::getline(file, line)) {
        trim(line);
        
        if (line.empty() || line[0] == ';' || line[0] == '#') {
            continue; // Skip comments and empty lines
        }
        
        if (line[0] == '[' && line.back() == ']') {
            current_section = line.substr(1, line.size() - 2);
            trim(current_section);
        } else {
            size_t equals_pos = line.find('=');
            if (equals_pos != std::string::npos) {
                std::string key = line.substr(0, equals_pos);
                std::string value = line.substr(equals_pos + 1);
                trim(key);
                trim(value);
                
                config_[current_section][key] = value;
            }
        }
    }
    
    return true;
}

std::string ConfigLoader::getString(const std::string& section, const std::string& key, 
                                   const std::string& default_value) {
    auto section_it = config_.find(section);
    if (section_it != config_.end()) {
        auto key_it = section_it->second.find(key);
        if (key_it != section_it->second.end()) {
            return key_it->second;
        }
    }
    return default_value;
}

int ConfigLoader::getInt(const std::string& section, const std::string& key, int default_value) {
    std::string value = getString(section, key);
    if (!value.empty()) {
        try {
            return std::stoi(value);
        } catch (...) {
            return default_value;
        }
    }
    return default_value;
}

std::string ConfigLoader::getAppDataPath(const std::string& app_name) {
    char app_data_path[MAX_PATH];
    if (SUCCEEDED(SHGetFolderPathA(NULL, CSIDL_APPDATA, NULL, 0, app_data_path))) {
        std::string path = std::string(app_data_path) + "\\" + app_name;
        return path;
    }
    return ""; // Fallback
}

void ConfigLoader::trim(std::string& str) {
    str.erase(str.begin(), std::find_if(str.begin(), str.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    str.erase(std::find_if(str.rbegin(), str.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), str.end());
}