#include "logger.h"
#include <windows.h>
#include <shlobj.h>
#include <sstream>
// 
#include <ctime>
#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <regex>

Logger::Logger(bool enable_console) : console_output(enable_console) {
    char exe_path[MAX_PATH];
    GetModuleFileNameA(NULL, exe_path, MAX_PATH);
    std::string exe_dir = std::filesystem::path(exe_path).parent_path().string();
    
    std::string log_dir = exe_dir + "\\logs";
    std::filesystem::create_directories(log_dir);
    
    log_file_path = getSessionFilename(log_dir);
    log_file.open(log_file_path, std::ios::app);
    
    if (log_file.is_open()) {
        std::string startup_msg = "=== Application Session Started ===";
        log_file << getCurrentTimestamp() << " " << startup_msg << "\n";
    }
}

Logger::~Logger() {
    if (log_file.is_open()) {
        std::string shutdown_msg = "=== Application Session Ended ===";
        log_file << getCurrentTimestamp() << " " << shutdown_msg << "\n";
        log_file.close();
    }
}

std::string Logger::getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::tm tm;
    localtime_s(&tm, &time_t);
    
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    ss << "." << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

std::string Logger::getSessionFilename(const std::string& log_dir) {
    
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::tm tm;
    localtime_s(&tm, &time_t);
    
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d");
    
    std::string date_prefix = ss.str();
    int max_session_number = 0;
    
    std::regex session_pattern(date_prefix + "-session-(\\d+)\\.txt");
    
    for (const auto& entry : std::filesystem::directory_iterator(log_dir)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            
            std::smatch matches;
            if (std::regex_match(filename, matches, session_pattern)) {
                // matches[1] содержит номер сессии
                try {
                    int session_num = std::stoi(matches[1].str());
                    if (session_num > max_session_number) {
                        max_session_number = session_num;
                    }
                } catch (...) {
                    // Игнорируем ошибки преобразования
                }
            }
        }
    }
    
    int session_number = max_session_number + 1;
    
    std::string filename = log_dir + "\\" + date_prefix + "-session-" + 
                          std::to_string(session_number) + ".txt";
    return filename;
}

void Logger::info(const std::string& message) {
    std::string log_entry = "[INFO] " + message;
    if (log_file.is_open()) {
        log_file << getCurrentTimestamp() << " " << log_entry << "\n";
        log_file.flush();
    }
    // if (console_output) {
    //     std::cout << log_entry << std::endl;
    // }
}

void Logger::error(const std::string& message) {
    std::string log_entry = "[ERROR] " + message;
    if (log_file.is_open()) {
        log_file << getCurrentTimestamp() << " " << log_entry << "\n";
        log_file.flush();
    }
    if (console_output) {
        std::cerr << log_entry << std::endl;
    }
}

void Logger::warning(const std::string& message) {
    std::string log_entry = "[WARNING] " + message;
    if (log_file.is_open()) {
        log_file << getCurrentTimestamp() << " " << log_entry << "\n";
        log_file.flush();
    }
    // if (console_output) {
    //     std::cout << log_entry << std::endl;
    // }
}

void Logger::debug(const std::string& message) {
    std::string log_entry = "[DEBUG] " + message;
    if (log_file.is_open()) {
        log_file << getCurrentTimestamp() << " " << log_entry << "\n";
        log_file.flush();
    }
}

std::string Logger::getLogFilePath() const {
    return log_file_path;
}