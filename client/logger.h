#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <iomanip>

class Logger {
private:
    std::ofstream log_file;
    std::string log_file_path;
    bool console_output;
    
    std::string getCurrentTimestamp();
    std::string getSessionFilename();

public:
    Logger(bool enable_console = false);
    ~Logger();
    
    void info(const std::string& message);
    void error(const std::string& message);
    void warning(const std::string& message);
    void debug(const std::string& message);
    
    std::string getLogFilePath() const;
};

#endif // LOGGER_H