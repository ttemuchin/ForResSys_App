cmake_minimum_required(VERSION 3.12)
project(MLApp)

set(CMAKE_CXX_STANDARD 17)

find_package(CURL REQUIRED)
find_package(JsonCpp REQUIRED)

add_executable(ResSys_app
    client/app.cpp
    client/http_client.cpp
    client/config_loader.cpp
)

target_include_directories(ResSys_app PRIVATE client)
target_link_libraries(ResSys_app PRIVATE ${CURL_LIBRARIES} JsonCpp::JsonCpp)

add_custom_command(TARGET ResSys_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/client/app_config.ini
        ${CMAKE_BINARY_DIR}/app_config.ini
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/python_server
        ${CMAKE_BINARY_DIR}/python_server
)