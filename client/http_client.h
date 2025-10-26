#ifndef HTTP_CLIENT_H
#define HTTP_CLIENT_H

#include <string>
#include <map>
#include <vector>
#include "logger.h"

class HttpClient {
public:
    HttpClient(const std::string& host, int port, int timeout_ms = 5000, Logger* logger = nullptr);
    ~HttpClient(); //деструктор;)

    // Основные методы
    std::string get(const std::string& endpoint);
    std::string post(const std::string& endpoint, const std::string& json_data);
    
    // Специальные методы сервера
    bool healthCheck();
    std::string trainModel(const std::string& base_name, const std::string& base_path, 
                      const std::string& config_path, const std::string& model_type);
    std::string predictWithModel(const std::string& file_path, const std::string& model_name, const std::string& base_name);
    
private:
    std::string host_;
    int port_;
    int timeout_ms_;
    Logger* logger_; 
    
    std::string buildUrl(const std::string& endpoint);
    static size_t writeCallback(void* contents, size_t size, size_t nmemb, std::string* response);
};

#endif // HTTP_CLIENT_H