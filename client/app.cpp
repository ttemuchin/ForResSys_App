#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <vector>
#include <sstream>
#include <algorithm>
#include "http_client.h"
#include "config_loader.h"
#include <windows.h>

namespace fs = std::filesystem;

std::string getExePath() {
    return std::filesystem::current_path().string();
}
std::string getParentExePath() {
    return std::filesystem::current_path().parent_path().string();
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
    std::string learning_base_dir_;

    std::vector<std::string> findLearningBases() {
        std::vector<std::string> bases;
        try {
            if (!fs::exists(learning_base_dir_)) {
                return bases;
            }
            
            for (const auto& entry : fs::directory_iterator(learning_base_dir_)) {
                if (entry.is_regular_file() && entry.path().extension() == ".dat") {
                    bases.push_back(entry.path().stem().string());
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error scanning learning bases: " << e.what() << std::endl;
        }
        return bases;
    }

    std::string getLearningBasePath(const std::string& base_name) {
        return learning_base_dir_ + "\\" + base_name + ".dat";
    }

    std::string getLearningBaseConfigPath(const std::string& base_name) {
        return learning_base_dir_ + "\\Configs\\" + base_name + ".txt";
    }

    
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
            
            learning_base_dir_ = config_.getAppDataPath() + "\\data\\LearningBase";
            
            std::cout << "Python path: " << python_path_ << std::endl;
            std::cout << "Server script: " << server_script_ << std::endl;
            std::cout << "Learning base dir: " << learning_base_dir_ << std::endl;

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
        std::cout << "Server stopped" << std::endl;
    }
    
    void makePrediction() {
        if (!http_client_.healthCheck()) {
            std::cerr << "Server is not available!" << std::endl;
            return;
        }
        
        std::cout << "Enter path to your data(.txt):" << std::endl;
        std::cout << "> ";
        std::string file_path;
        std::getline(std::cin, file_path);
        
        if (!fs::exists(file_path)) {
            std::cerr << "File not found: " << file_path << std::endl;
            return;
        }
        
        std::cout << "Choose model:" << std::endl;
        std::cout << "1. SVR" << std::endl;
        std::cout << "2. Convolutional Layers" << std::endl;
        std::cout << "3. Linear Regression" << std::endl;
        std::cout << "Enter number: ";
        
        std::string model_choice;
        std::getline(std::cin, model_choice);
        
        std::string model_name;
        if (model_choice == "1") {
            model_name = "svr";
        } else if (model_choice == "2") {
            model_name = "convolutional";
        } else if (model_choice == "3") {
            model_name = "linear_regression";
        } else {
            std::cout << "Invalid model choice!" << std::endl;
            return;
        }
        
        std::cout << "Making prediction for file: " << file_path << " with model: " << model_name << std::endl;
        
        std::string result = http_client_.predictWithModel(file_path, model_name);
        
        std::cout << "Results saved to file - " << result << std::endl;
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

    struct LearningBaseConfig {
        std::string name;
        int num_samples;
        int num_targets_y;
        std::vector<double> y_precision;
        int num_features_x;
        std::vector<int> x_lengths;
    };

    bool parseLearningBaseConfig(const std::string& input, LearningBaseConfig& config) {
        std::vector<std::string> tokens;
        std::istringstream iss(input);
        std::string token;
        
        while (iss >> token) {
            tokens.push_back(token);
        }
        
        if (tokens.size() < 4) {
            std::cerr << "Error: Not enough parameters" << std::endl;
            return false;
        }
        
        try {
            // Collect all tokens
            config.name = tokens[0];
            
            config.num_samples = std::stoi(tokens[1]);
            
            config.num_targets_y = std::stoi(tokens[2]);
            
            if (tokens.size() < 3 + config.num_targets_y + 1) {
                std::cerr << "Error: Not enough precision values for Y" << std::endl;
                return false;
            }
            
            config.y_precision.clear();
            for (int i = 0; i < config.num_targets_y; ++i) {
                config.y_precision.push_back(std::stod(tokens[3 + i]));
            }
            
            int current_index = 3 + config.num_targets_y;
            if (tokens.size() <= current_index) {
                std::cerr << "Error: Missing number of features X" << std::endl;
                return false;
            }
            
            config.num_features_x = std::stoi(tokens[current_index]);
            current_index++;
            
            if (tokens.size() < current_index + config.num_features_x) {
                std::cerr << "Error: Not enough length values for X" << std::endl;
                return false;
            }
            
            config.x_lengths.clear();
            for (int i = 0; i < config.num_features_x; ++i) {
                config.x_lengths.push_back(std::stoi(tokens[current_index + i]));
            }
            
            if (tokens.size() != current_index + config.num_features_x) {
                std::cerr << "Error: Too many parameters provided" << std::endl;
                return false;
            }
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Error parsing parameters: " << e.what() << std::endl;
            return false;
        }
    }

    bool copyLearningBaseFile(const std::string& source_path, const std::string& base_name) {
        try {
            fs::create_directories(learning_base_dir_);
            
            std::string dest_path = learning_base_dir_ + "\\" + base_name + ".txt";
            
            fs::copy_file(source_path, dest_path, fs::copy_options::overwrite_existing);
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error copying file: " << e.what() << std::endl;
            return false;
        }
    }

    bool saveLearningBaseConfig(const LearningBaseConfig& config) {
        try {
            std::string config_dir = learning_base_dir_ + "\\Configs";
            fs::create_directories(config_dir);
            
            std::string config_path = config_dir + "\\" + config.name + ".txt";
            std::ofstream file(config_path);
            
            if (!file.is_open()) {
                return false;
            }
            
            file << "name=" << config.name << "\n";
            file << "num_samples=" << config.num_samples << "\n";
            file << "num_targets_y=" << config.num_targets_y << "\n";
            
            file << "y_precision=";
            for (size_t i = 0; i < config.y_precision.size(); ++i) {
                file << config.y_precision[i];
                if (i < config.y_precision.size() - 1) file << ",";
            }
            file << "\n";
            
            file << "num_features_x=" << config.num_features_x << "\n";
            
            file << "x_lengths=";
            for (size_t i = 0; i < config.x_lengths.size(); ++i) {
                file << config.x_lengths[i];
                if (i < config.x_lengths.size() - 1) file << ",";
            }
            file << "\n";
            
            file.close();
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Error saving config: " << e.what() << std::endl;
            return false;
        }
    }

    void uploadLearningBase() {
        LearningBaseConfig config;
        
        std::cout << "Input config of the learning base:" << std::endl;
        std::cout << "Format: name num_samples num_targets_y y_precision1 y_precision2 ... num_features_x x_length1 x_length2 ..." << std::endl;
        std::cout << "Example: Base123 1000 2 0.01 0.05 3 256 128 64" << std::endl;
        std::cout << "> ";
        
        std::string config_input;
        std::getline(std::cin, config_input);
        
        if (!parseLearningBaseConfig(config_input, config)) {
            std::cerr << "Invalid configuration format!" << std::endl;
            return;
        }
        
        std::cout << "OK" << std::endl;
        
        std::cout << "Input learning base (path to file):" << std::endl;
        std::cout << "Example: C:/Users/ttemuchin4/Downloads/Telegram Desktop/Landing.txt" << std::endl;
        std::cout << "> ";
        
        std::string file_path;
        std::getline(std::cin, file_path);
        
        if (!fs::exists(file_path)) {
            std::cerr << "File not found: " << file_path << std::endl;
            return;
        }
        
        if (!copyLearningBaseFile(file_path, config.name)) {
            std::cerr << "Failed to copy learning base file!" << std::endl;
            return;
        }
        
        if (!saveLearningBaseConfig(config)) {
            std::cerr << "Failed to save learning base configuration!" << std::endl;
            return;
        }
        
        std::cout << "Learning base " << config.name << " saved successfully!" << std::endl;
        std::cout << "Name: " << config.name << std::endl;
        
    }

    ///////////
    void startLearning() {
        auto bases = findLearningBases();
        
        if (bases.empty()) {
            std::cout << "No learning bases found. Please upload a base first." << std::endl;
            return;
        }
        
        std::cout << "Choose the learning base:" << std::endl;
        for (size_t i = 0; i < bases.size(); ++i) {
            std::cout << (i + 1) << ". " << bases[i] << std::endl;
        }
        
        std::cout << "Enter number: ";
        std::string choice_str;
        std::getline(std::cin, choice_str);
        
        int base_choice;
        try {
            base_choice = std::stoi(choice_str);
            if (base_choice < 1 || base_choice > static_cast<int>(bases.size())) {
                std::cout << "Invalid choice!" << std::endl;
                return;
            }
        } catch (...) {
            std::cout << "Invalid number!" << std::endl;
            return;
        }
        
        std::string selected_base = bases[base_choice - 1];
        std::string base_path = getLearningBasePath(selected_base);
        std::string config_path = getLearningBaseConfigPath(selected_base);
        
        if (!fs::exists(base_path) || !fs::exists(config_path)) {
            std::cout << "Selected base files not found!" << std::endl;
            return;
        }
        
        std::cout << "Choose model:" << std::endl;
        std::cout << "1. SVR" << std::endl;
        std::cout << "2. Convolutional Layers" << std::endl;
        std::cout << "3. Linear Regression" << std::endl;
        std::cout << "Enter number: ";
        
        std::string model_choice_str;
        std::getline(std::cin, model_choice_str);
        
        std::string model_name;
        if (model_choice_str == "1") {
            model_name = "svr";
        } else if (model_choice_str == "2") {
            model_name = "convolutional";
        } else if (model_choice_str == "3") {
            model_name = "linear_regression";
        } else {
            std::cout << "Invalid model choice!" << std::endl;
            return;
        }
        
        std::cout << "Start Learning? y/n: ";
        std::string confirm;
        std::getline(std::cin, confirm);
        
        if (confirm != "y" && confirm != "Y") {
            std::cout << "Learning cancelled." << std::endl;
            return;
        }
        
        if (!http_client_.healthCheck()) {
            std::cout << "Server is not available. Please start the server first." << std::endl;
            return;
        }
        
        std::cout << "Starting learning process..." << std::endl;
        
        std::string response = http_client_.trainModel(selected_base, base_path, config_path, model_name);
        std::cout << "Server response: " << response << std::endl;
    }
    
    void showMenu() {
        std::cout << "\n=== ML Application ===" << std::endl;
        std::cout << "1. Start server" << std::endl;
        std::cout << "2. Upload learning base" << std::endl;
        std::cout << "3. Start learning" << std::endl;
        std::cout << "4. Make prediction" << std::endl;
        std::cout << "5. Check server health" << std::endl;
        std::cout << "6. Stop server" << std::endl;
        std::cout << "7. Exit" << std::endl;
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
                uploadLearningBase();
            } else if (choice == "3") {
                startLearning();
            } else if (choice == "4") {
                makePrediction();
            } else if (choice == "5") {
                if (http_client_.healthCheck()) {
                    std::cout << "✓ Server is healthy!" << std::endl;
                } else {
                    std::cout << "✗ Server is not available!" << std::endl;
                }
            } else if (choice == "6") {
                stopServerSoft();
            } else if (choice == "7") {
                break;
            } else {
                std::cout << "Invalid option!" << std::endl;
            }
        }
        
        stopServer();
        std::cout << "Application closed" << std::endl;
    }
};

int main() {
    SetConsoleOutputCP(CP_UTF8);
    MLApplication app;
    app.run();
    return 0;
}