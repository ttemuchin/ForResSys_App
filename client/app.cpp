#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <vector>
#include <sstream>
#include <algorithm>
#include <json/json.h>
#include <json/value.h>
#include <json/reader.h>
#include "http_client.h"
#include "config_loader.h"
#include "logger.h"
#include <windows.h>

namespace fs = std::filesystem;

std::string getExePath() {
    return std::filesystem::current_path().string();
}

class MLApplication {
private:
    Logger logger_;
    ConfigLoader config_;
    HttpClient http_client_;
    
    std::thread server_thread_;
    bool server_running_ = false;
    
    std::string app_base_dir_;
    std::string python_path_;
    std::string server_script_;
    std::string learning_base_dir_;

public:
    MLApplication()
    : logger_(false),
      config_(getExePath() + "\\app_config.ini"),  //full path to config 
      http_client_("localhost", 8000, 300000, &logger_)
      
    {
        app_base_dir_ = getExePath();
        logger_.info("EXE directory: " + app_base_dir_);

        if (config_.load()) {
            python_path_ = app_base_dir_ + "\\" + config_.getString("paths", "python_path");
            server_script_ = app_base_dir_ + "\\" + config_.getString("paths", "server_script");
            learning_base_dir_ = app_base_dir_+ "\\data\\LearningBase";
            
            logger_.info("Python server: " + python_path_);
            logger_.info("Server script: " + server_script_);
            logger_.info("Learning base dir: " + learning_base_dir_);

            createDirectories();

        } else {
            logger_.error("Failed to load config from: " + getExePath() + "\\app_config.ini");
        }
    }

private:
    void createDirectories() {
        std::vector<std::string> dirs = {
            learning_base_dir_ + "\\Configs",
            app_base_dir_ + "\\output",
            app_base_dir_ + "\\logs",
            app_base_dir_ + "\\models"
        };
        
        for (const auto& dir : dirs) {
            fs::create_directories(dir);
        }
    }

    void showHelp() {
        std::cout << "\n=== Available commands: ===" << std::endl;
        std::cout << "  help             - Show this help message" << std::endl;
        std::cout << "  view bases       - List available(saved) training bases" << std::endl;
        std::cout << "  json help        - Show JSON configuration examples" << std::endl;
        std::cout << "  exit (quit or q) - Exit application" << std::endl;
        std::cout << "\nBasic usage:" << std::endl;
        std::cout << "  1. Create a JSON configuration file" << std::endl;
        std::cout << "  2. Enter path to JSON file when prompted" << std::endl;
        std::cout << "  3. Server will process train/predict automatically" << std::endl;
    }

    void showJsonHelp() {
        std::cout << "\n=== JSON Configuration Examples ===" << std::endl;
        
        std::cout << "\n1. Training (e.g. train.json):" << std::endl;
        std::cout << R"({
  "method": "train",
  "model": "convolutional",
  "baseConfig": {
    "name": "BaseTEST5",
    "N": 100,
    "nY": 2,
    "accuracy": [0.001, 0.005],
    "nX": 2,
    "dimension": [400, 60]
  },
  "basePath": "C:/path/to/training_data.txt"
})" << std::endl;
        
        std::cout << "\n2. Prediction (e.g. predict.json):" << std::endl;
        std::cout << R"({
  "method": "predict",
  "model": "convolutional",
  "baseName": "BaseTEST5",
  "predPath": "C:/path/to/prediction_data.txt"
})" << std::endl;
        
        std::cout << "\nSupported models: svr, convolutional, linear_regression" << std::endl;
    }

    void viewBases() {
        std::vector<std::string> bases = findLearningBases();
        
        if (bases.empty()) {
            std::cout << "No training bases found." << std::endl;
            std::cout << "Use 'train' method in JSON to create a new base." << std::endl;
            return;
        }
        
        std::cout << "\n=== Available Training Bases ===" << std::endl;
        for (size_t i = 0; i < bases.size(); ++i) {
            std::cout << "  " << (i + 1) << ". " << bases[i] << std::endl;
            
            // ! ВАЖНО ! т к в дальнейшем нужно будет вытащить из JSON только поле с количеством семплов
            std::string config_path = learning_base_dir_ + "\\Configs\\" + bases[i] + ".txt";
            if (fs::exists(config_path)) {
                std::ifstream config_file(config_path);
                std::string line;
                std::cout << "     Config: ";
                while (std::getline(config_file, line)) {
                    std::cout << line << " ";
                }
                std::cout << std::endl;
            }
        }
    }
    
    std::vector<std::string> findLearningBases() {
        std::vector<std::string> bases;
        try {
            if (!fs::exists(learning_base_dir_)) {
                return bases;
            }
            
            for (const auto& entry : fs::directory_iterator(learning_base_dir_)) {
                if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                    bases.push_back(entry.path().stem().string());
                }
            }
        } catch (const std::exception& e) {
            logger_.error("Error scanning learning bases: " + std::string(e.what()));
        }
        return bases;
    }

    std::string readJsonFile(const std::string& file_path) {
        try {
            std::ifstream file(file_path);
            if (!file.is_open()) {
                return "";
            }
            
            std::stringstream buffer;
            buffer << file.rdbuf();
            return buffer.str();
            
        } catch (const std::exception& e) {
            logger_.error("Error reading JSON file: " + std::string(e.what()));
            return "";
        }
    }

    bool sendJsonToServer(const std::string& json_file_path) {
        if (!fs::exists(json_file_path)) {
            std::cout << "✗ JSON file not found: " << json_file_path << std::endl;
            return false;
        }
        
        std::string json_content = readJsonFile(json_file_path);
        if (json_content.empty()) {
            std::cout << "✗ Failed to read JSON file or file is empty" << std::endl;
            return false;
        }
        
        if (!http_client_.healthCheck()) {
            std::cout << "✗ Server is not available. Please start the server first." << std::endl;
            return false;
        }
        
        std::cout << "\nSending JSON request to server..." << std::endl;
        logger_.info("Sending JSON request from file: " + json_file_path);
        
        Json::Value request_json;
        Json::Reader reader;
        
        if (!reader.parse(json_content, request_json)) {
            std::cout << "✗ Invalid JSON format" << std::endl;
            return false;
        }
        
        Json::Value wrapper;
        wrapper["json_data"] = request_json;
        
        Json::StreamWriterBuilder writer;
        std::string json_request = Json::writeString(writer, wrapper);
        
        std::string response = http_client_.post("/process_json", json_request); // only one route
        logger_.info("Server response: " + response);
        
        Json::Value response_json;
        Json::Reader response_reader;
        
        if (response_reader.parse(response, response_json)) {
            if (response_json["status"].asString() == "error" || 
                response_json.isMember("status") && response_json["status"].asString() == "error") {
                std::cout << "✗ Server returned error: " << response_json["message"].asString() << std::endl;
                return false;
            }
            
            // processing response
            if (response_json.isMember("operation")) {
                std::string operation = response_json["operation"].asString();
                
                if (operation == "train") {
                    if (response_json.isMember("train_result") && 
                        response_json["train_result"]["status"].asString() == "success") {
                        std::cout << "  Training completed successfully!" << std::endl;
                        auto train_result = response_json["train_result"];
                        std::cout << "  R² score: " << train_result["best_r2"].asFloat() << std::endl;
                        std::cout << "  Best loss: " << train_result["best_loss"].asFloat() << std::endl;
                        std::cout << "  Weights saved to: " << train_result["weights_path"].asString() << std::endl;
                        return true;
                    }
                } else if (operation == "predict") {
                    if (response_json.isMember("predict_result") && 
                        response_json["predict_result"]["status"].asString() == "success") {
                        std::cout << "  Prediction completed successfully!" << std::endl;
                        auto predict_result = response_json["predict_result"];
                        std::cout << "  Results saved to: " << predict_result["output_path"].asString() << std::endl;
                        return true;
                    }
                }
            }
        }
        
        std::cout << "✓ Request processed by server" << std::endl;
        std::cout << "Response: " << response << std::endl;
        return true;
    }


public:
    void startServer() {
        if (!fs::exists(python_path_) || !fs::exists(server_script_)) {
            logger_.error("Python server files not found!");
            return;
        }
        
        server_running_ = true;
        server_thread_ = std::thread([this]() {
            std::string log_path = logger_.getLogFilePath();
            size_t last_dot = log_path.find_last_of('.');
            std::string python_log_path = log_path.substr(0, last_dot) + "_PyServer.txt";

            std::string command = "cd /d \"" + 
                         fs::path(server_script_).parent_path().string() + 
                         "\" && \"" + python_path_ + "\" \"" + server_script_ + "\""+
                         ">> \"" + python_log_path + "\" 2>&1";  // ПЕРЕНАПРАВЛЯЕМ ВСЕ В отдельный ЛОГ-файл
            
            logger_.debug("Executing server command: " + command);
            logger_.info("Python server logs will be saved to: " + python_log_path);
            std::system(command.c_str());
        });
        
        std::cout << "Starting Python server..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(3));
        
        if (waitForServer(10)) {
            std::cout << "✓ Server started successfully!" << std::endl;
            logger_.info("Python server started successfully on localhost:8000");
        } else {
            std::cerr << "✗ Server failed to start!" << std::endl;
            logger_.error("Python server failed to start");
            server_running_ = false;
        }
    }
    
    bool waitForServer(int max_attempts) {
        for (int i = 0; i < max_attempts; ++i) {
            if (http_client_.healthCheck()) {
                logger_.info("Server health check passed");
                return true;
            }

            std::cout << "Waiting for server... (" << i+1 << "/" << max_attempts << ")" << std::endl;
            logger_.debug("Waiting for server... Attempt " + std::to_string(i+1) + "/" + std::to_string(max_attempts));
            std::this_thread::sleep_for(std::chrono::seconds(1));

        }
        logger_.error("Server health check failed after " + std::to_string(max_attempts) + " attempts");
        return false;
    }
    
    void stopServerSoft() {
        server_running_ = false;
        logger_.info("Attempting graceful server shutdown");
        
        HttpClient temp_client("localhost", 8000, 1000);
        try {
            temp_client.post("/shutdown", "");
            logger_.info("Graceful shutdown signal sent to server");
        } catch (...) {
            logger_.warning("Failed to send graceful shutdown signal");
        }
        
        if (server_thread_.joinable()) {
            server_thread_.join();
        }
        logger_.info("Server stopped gracefully");
    }

    void stopServer() {
        server_running_ = false;
        
        #ifdef _WIN32
            system("taskkill /f /im python.exe 2>nul 1>nul");
        #else
            system("pkill -f python 2>/dev/null");
        #endif
        
        if (server_thread_.joinable()) {
            server_thread_.detach();
        }
        std::cout << "Server stopped" << std::endl;
    }
    
    void run() {
        logger_.info("ResSysML Application v2.2 started");
        std::cout << "=== ResSysML v2.2 ===" << std::endl;
        std::cout << "Type 'help' for available commands" << std::endl;
        
        startServer();
        
        while (true) {
            std::cout << "\nEnter command or JSON file path: ";
            std::string input;
            std::getline(std::cin, input);
            
            if (input.empty()) {
                continue;
            }
            
            // Команды
            if (input == "help" || input == "?") {
                showHelp();
                continue;
            } 
            else if (input == "view bases") {
                viewBases();
                continue;
            }
            else if (input == "json help") {
                showJsonHelp();
                continue;
            }
            else if (input == "exit" || input == "quit" || input == "q") {
                break;
            }

            bool success = sendJsonToServer(input);
        
            if (success) {
                std::cout << "\n✓ Operation completed successfully!" << std::endl;
            } else {
                std::cout << "\n✗ Operation failed. Check logs for details." << std::endl;
            }
        }
        
        stopServerSoft();
        std::cout << "Application closed" << std::endl;
        logger_.info("Application session ended");
    }
};

int main() {
    SetConsoleOutputCP(CP_UTF8);
    MLApplication app;
    app.run();
    return 0;
}