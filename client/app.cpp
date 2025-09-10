#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <fstream>
#include <filesystem>
#include "http_client.h"
#include "config_loader.h"
#include <windows.h>

namespace fs = std::filesystem;

std::string getExePath() {
    return std::filesystem::current_path().string();
}

class MLApplication {
private:
    ConfigLoader config_;
    HttpClient http_client_;
    std::thread server_thread_;
    bool server_running_ = false;
    
    std::string python_path_;
    std::string server_script_;
    std::string output_file_;
    
public:
    MLApplication()
    : config_(getExePath() + "\\app_config.ini"),  //full path to config
      http_client_("localhost", 8000, 5000) {
        
        std::string exeDir = getExePath();
        std::cout << "EXE directory: " << exeDir << std::endl;
        
        if (config_.load()) {
            python_path_ = exeDir + "\\" + config_.getString("paths", "python_path");
            server_script_ = exeDir + "\\" + config_.getString("paths", "server_script");
            output_file_ = exeDir + "\\" + config_.getString("paths", "output_file");
            
            std::cout << "Python path: " << python_path_ << std::endl;
            std::cout << "Server script: " << server_script_ << std::endl;
        } else {
            std::cerr << "Failed to load config from: " << getExePath() + "\\app_config.ini" << std::endl;
        }
    }

    void startServer() {
        if (!fs::exists(python_path_) || !fs::exists(server_script_)) {
            std::cerr << "Python server files not found!" << std::endl;
            return;
        }
        
        server_running_ = true;
        server_thread_ = std::thread([this]() {
            std::string command = "cd /d \"" + 
                         fs::path(server_script_).parent_path().string() + 
                         "\" && \"" + python_path_ + "\" \"" + server_script_ + "\"";
            std::system(command.c_str());
        });
        
        std::cout << "Starting Python server..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(3));
        
        if (waitForServer(10)) {
            std::cout << "✓ Server started successfully!" << std::endl;
        } else {
            std::cerr << "✗ Server failed to start!" << std::endl;
            server_running_ = false;
        }
    }
    
    bool waitForServer(int max_attempts) {
        for (int i = 0; i < max_attempts; ++i) {
            if (http_client_.healthCheck()) {
                return true;
            }
            std::cout << "Waiting for server... (" << i+1 << "/" << max_attempts << ")" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        return false;
    }
    
    void stopServerSoft() {
        server_running_ = false;
        
        HttpClient temp_client("localhost", 8000, 1000);
        try {
            temp_client.post("/shutdown", "");
        } catch (...) {
            //
        }
        
        if (server_thread_.joinable()) {
            server_thread_.join();
        }
        std::cout << "Server stopped(soft)." << std::endl;
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
        std::cout << "Server stopped(hard)." << std::endl;
    }
    
    void makePrediction(const std::string& input) {
        if (!http_client_.healthCheck()) {
            std::cerr << "Server is not available!" << std::endl;
            return;
        }
        
        std::cout << "Making prediction for: " << input << std::endl;
        std::string result = http_client_.predict(input);
        
        saveResultToFile(result);
        std::cout << "✓ Prediction saved to: " << output_file_ << std::endl;
        std::cout << "Result: " << result << std::endl;
    }
    
    void saveResultToFile(const std::string& result) {
        fs::path output_path(output_file_);
        fs::create_directories(output_path.parent_path());
        
        std::ofstream file(output_file_);
        if (file.is_open()) {
            file << "Prediction Result:\n";
            file << "==================\n";
            file << result << "\n";
            file << "==================\n";
            file.close();
        }
    }
    
    void showMenu() {
        std::cout << "\n=== ML Application ===" << std::endl;
        std::cout << "1. Start server" << std::endl;
        std::cout << "2. Make prediction" << std::endl;
        std::cout << "3. Check server health" << std::endl;
        std::cout << "4. Exit" << std::endl;
        std::cout << "5. Stop server" << std::endl;
        std::cout << "Choose option: ";
    }
    
    void run() {
        std::cout << "ML Desktop Application v1.0" << std::endl;
        
        while (true) {
            showMenu();
            
            std::string choice;
            std::getline(std::cin, choice);
            
            if (choice == "1") {
                startServer();
            } else if (choice == "2") {
                std::string input;
                std::cout << "Enter input data: ";
                std::getline(std::cin, input);
                makePrediction(input);
            } else if (choice == "3") {
                if (http_client_.healthCheck()) {
                    std::cout << "✓ Server is healthy!" << std::endl;
                } else {
                    std::cout << "✗ Server is not available!" << std::endl;
                }
            } else if (choice == "4") {
                break;
            } else if (choice == "5") {
                stopServerSoft();
            } else {
                std::cout << "Invalid option!" << std::endl;
            }
        }
        
        stopServer();
        std::cout << "Application closed." << std::endl;
    }
};

int main() {
    SetConsoleOutputCP(CP_UTF8);
    MLApplication app;
    app.run();
    return 0;
}