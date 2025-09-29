#ifndef CONFIG_LOADER_H
#define CONFIG_LOADER_H

#include <string>
#include <map>

class ConfigLoader {
public:
    ConfigLoader(const std::string& filename);
    
    bool load();
    std::string getAppDataPath(const std::string& app_name = "ResSysApp");

    std::string getString(const std::string& section, const std::string& key, 
                         const std::string& default_value = "");
    int getInt(const std::string& section, const std::string& key, int default_value = 0);
    
private:
    std::string filename_;
    std::map<std::string, std::map<std::string, std::string>> config_;
    
    void trim(std::string& str);
};

#endif // CONFIG_LOADER_H