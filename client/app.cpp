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
    
    std::string python_path_;
    std::string server_script_;
    std::string output_file_;
    std::string learning_base_dir_;

    struct JobConfig {
        std::string base_config;// "BaseName 100 2 0.001 0.005 2 400 60"
        std::string training_file_path;
        std::string model_name;
        std::string prediction_file_path;
    };

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
            std::cerr << "Error scanning learning bases: " << e.what() << std::endl;
        }
        return bases;
    }

    std::string getLearningBasePath(const std::string& base_name) {
        return learning_base_dir_ + "\\" + base_name + ".txt";
    }

    std::string getLearningBaseConfigPath(const std::string& base_name) {
        return learning_base_dir_ + "\\Configs\\" + base_name + ".txt";
    }

    bool parseJobConfig(const std::string& config_file_path, JobConfig& job_config) {
        try {
            std::ifstream file(config_file_path);
            if (!file.is_open()) {
                logger_.error("Cannot open config file: " + config_file_path);
                return false;
            }
            
            std::vector<std::string> lines;
            std::string line;
            while (std::getline(file, line)) {
                if (!line.empty() && line[0] != '#') {
                    lines.push_back(line);
                }
            }
            file.close();
            
            if (lines.size() < 4) {
                logger_.error("Config file must contain exactly 4 lines");
                return false;
            }
            
            job_config.base_config = lines[0];
            job_config.training_file_path = lines[1];
            job_config.model_name = lines[2];
            job_config.prediction_file_path = lines[3];
            
            logger_.info("Job config parsed successfully");
            logger_.info("Base config: " + job_config.base_config);
            logger_.info("Training file: " + job_config.training_file_path);
            logger_.info("Model: " + job_config.model_name);
            logger_.info("Prediction file: " + job_config.prediction_file_path);
            
            return true;
            
        } catch (const std::exception& e) {
            logger_.error("Error parsing job config: " + std::string(e.what()));
            return false;
        }
    }
    
public:
    MLApplication()
    : logger_(false),
      config_(getExePath() + "\\app_config.ini"),  //full path to config 
      http_client_("localhost", 8000, 300000, &logger_)
      
    {
        std::string exeDir = getExePath();
        logger_.info("EXE directory: " + exeDir);

        if (config_.load()) {
            python_path_ = exeDir + "\\" + config_.getString("paths", "python_path");
            server_script_ = exeDir + "\\" + config_.getString("paths", "server_script");
            output_file_ = exeDir + "\\" + config_.getString("paths", "output_file");
            
            learning_base_dir_ = config_.getAppDataPath() + "\\data\\LearningBase";
            
            logger_.info("Python path: " + python_path_);
            logger_.info("Server script: " + server_script_);
            logger_.info("Learning base dir: " + learning_base_dir_);

        } else {
            logger_.error("Failed to load config from: " + getExePath() + "\\app_config.ini");
        }
    }

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
    
    bool makePrediction(const std::string& prediction_file_path, const std::string& model_name, const std::string& base_name) {
        if (!http_client_.healthCheck()) {
            logger_.error("Server not available for prediction");
            return false;
        }
        
        if (!fs::exists(prediction_file_path)) {
            logger_.error("Prediction file not found: " + prediction_file_path);
            return false;
        }
        
        std::cout << "Making prediction..." << std::endl;
        logger_.info("Starting prediction - File: " + prediction_file_path + ", Base: " + base_name + ", Model: " + model_name);
        
        std::string result = http_client_.predictWithModel(prediction_file_path, model_name, base_name);
        logger_.info("Prediction server response: " + result);
        
        // Проверяем успешность предсказания по ответу сервера
        if (result.find("\"status\":\"success\"") != std::string::npos) {
            // Извлекаем путь к результату из JSON ответа
            size_t path_start = result.find("\"output_path\":\"");
            if (path_start != std::string::npos) {
                path_start += 25; // Длина "\"output_path\":\""
                size_t path_end = result.find("\"", path_start);
                if (path_end != std::string::npos) {
                    std::string output_path = result.substr(path_start, path_end - path_start);
                    std::cout << "✓ Results saved to: " << output_path << std::endl;
                }
            }
            return true;
        } else {
            std::cout << "✗ Prediction failed!" << std::endl;
            return false;
        }
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

    bool uploadLearningBase(const std::string& base_config, const std::string& training_file_path) {
        LearningBaseConfig config;
        
        if (!parseLearningBaseConfig(base_config, config)) {
            logger_.error("Failed to parse learning base config: " + base_config);
            return false;
        }
        
        logger_.info("Learning base config parsed successfully: " + config.name);
        
        if (!fs::exists(training_file_path)) {
            logger_.error("Learning base file not found: " + training_file_path);
            return false;
        }
        
        if (!copyLearningBaseFile(training_file_path, config.name)) {
            logger_.error("Failed to copy learning base file: " + training_file_path + " to " + config.name);
            return false;
        }
        
        if (!saveLearningBaseConfig(config)) {
            logger_.error("Failed to save learning base config for: " + config.name);
            return false;
        }
        
        std::cout << "✓ Learning base " << config.name << " uploaded successfully!" << std::endl;
        logger_.info("Learning base uploaded successfully: " + config.name);
        return true;
    }

    ///////////
    bool startLearning(const std::string& base_name, const std::string& model_name) {
        std::string base_path = getLearningBasePath(base_name);
        std::string config_path = getLearningBaseConfigPath(base_name);
        
        if (!fs::exists(base_path) || !fs::exists(config_path)) {
            logger_.error("Selected learning base files not found: " + base_name);
            return false;
        }
        
        if (!http_client_.healthCheck()) {
            logger_.error("Server not available for training");
            return false;
        }
        
        std::cout << "Starting learning process..." << std::endl;
        logger_.info("Starting training - Base: " + base_name + ", Model: " + model_name);
        
        std::string response = http_client_.trainModel(base_name, base_path, config_path, model_name);
        logger_.info("Training server response: " + response);
        
        // Проверяем успешность обучения по ответу сервера
        if (response.find("\"status\":\"success\"") != std::string::npos) {
            std::cout << "✓ Training completed successfully!" << std::endl;
            return true;
        } else {
            std::cout << "✗ Training failed!" << std::endl;
            return false;
        }
    }

    bool executeJob(const std::string& config_file_path) {
        JobConfig job_config;
        
        if (!parseJobConfig(config_file_path, job_config)) {
            return false;
        }
        
        LearningBaseConfig base_config_obj;
        if (!parseLearningBaseConfig(job_config.base_config, base_config_obj)) {
            return false;
        }
        
        std::cout << "\n=== Starting ML Pipeline ===" << std::endl;
        std::cout << "Base: " << base_config_obj.name << std::endl;
        std::cout << "Model: " << job_config.model_name << std::endl;
        std::cout << "Training file: " << job_config.training_file_path << std::endl;
        std::cout << "Prediction file: " << job_config.prediction_file_path << std::endl;
        
        // 1. Загружаем обучающую базу
        std::cout << "\n1. Uploading learning base..." << std::endl;
        if (!uploadLearningBase(job_config.base_config, job_config.training_file_path)) {
            return false;
        }
        
        // 2. Обучаем модель
        std::cout << "\n2. Training model..." << std::endl;
        if (!startLearning(base_config_obj.name, job_config.model_name)) {
            return false;
        }
        
        // 3. Делаем предсказание
        std::cout << "\n3. Making prediction..." << std::endl;
        if (!makePrediction(job_config.prediction_file_path, job_config.model_name, base_config_obj.name)) {
            return false;
        }
        
        std::cout << "\n✓ Pipeline completed successfully!" << std::endl;
        return true;
    }
    
    void run() {
        logger_.info("ML Desktop Application v2.1 started");
        std::cout << "ML Desktop Application v2.1" << std::endl;
        
        startServer();
        
        while (true) {
            std::cout << "\nEnter path to job config file (or 'exit' to quit):" << std::endl;
            std::cout << "> ";
            
            std::string input;
            std::getline(std::cin, input);
            
            if (input == "exit" || input == "quit") {
                break;
            }
            
            if (!fs::exists(input)) {
                std::cout << "✗ Config file not found: " << input << std::endl;
                continue;
            }
            
            bool success = executeJob(input);
            
            if (success) {
                std::cout << "\n✓ Job completed successfully!" << std::endl;
            } else {
                std::cout << "\n✗ Job failed! Check logs for details." << std::endl;
            }
            
            std::cout << "\nNew predict? (y/n): ";
            std::string answer;
            std::getline(std::cin, answer);
            
            if (answer != "y" && answer != "Y") {
                break;
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