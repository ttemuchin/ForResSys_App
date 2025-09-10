#include "http_client.h"
#include <curl/curl.h>
#include <iostream>
#include <sstream>
#include <json/json.h>

HttpClient::HttpClient(const std::string& host, int port, int timeout_ms)
    : host_(host), port_(port), timeout_ms_(timeout_ms) {
    curl_global_init(CURL_GLOBAL_DEFAULT);
}

HttpClient::~HttpClient() {
    curl_global_cleanup();
}

std::string HttpClient::buildUrl(const std::string& endpoint) {
    return "http://" + host_ + ":" + std::to_string(port_) + endpoint;
}

size_t HttpClient::writeCallback(void* contents, size_t size, size_t nmemb, std::string* response) {
    size_t totalSize = size * nmemb;
    response->append((char*)contents, totalSize);
    return totalSize;
}

std::string HttpClient::get(const std::string& endpoint) {
    CURL* curl = curl_easy_init();
    std::string response;
    
    if (curl) {
        std::string url = buildUrl(endpoint);
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeout_ms_);
        
        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            std::cerr << "HTTP GET failed: " << curl_easy_strerror(res) << std::endl;
        }
        
        curl_easy_cleanup(curl);
    }
    
    return response;
}

std::string HttpClient::post(const std::string& endpoint, const std::string& json_data) {
    CURL* curl = curl_easy_init();
    std::string response;
    
    if (curl) {
        std::string url = buildUrl(endpoint);
        
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_data.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, json_data.length());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeout_ms_);
        
        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            std::cerr << "HTTP POST failed: " << curl_easy_strerror(res) << std::endl;
        }
        
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }
    
    return response;
}

bool HttpClient::healthCheck() {
    try {
        std::string response = get("/health");
        Json::Value json;
        Json::Reader reader;
        
        if (reader.parse(response, json)) {
            return json["status"].asString() == "healthy" && 
                   json["model_loaded"].asBool();
        }
    } catch (...) {
        // Ignore errors, return false
    }
    return false;
}

std::string HttpClient::predict(const std::string& input_data) {
    Json::Value request;
    request["input_data"] = input_data;
    
    Json::StreamWriterBuilder writer;
    std::string json_request = Json::writeString(writer, request);
    
    return post("/predict", json_request);
}