# The following lines of boilerplate have to be in your project's
# CMakeLists in this exact order for cmake to work correctly
cmake_minimum_required(VERSION 3.16)

# Pull in ESP-IDF
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

project(idp_firmware)

# C++17 required for TFLM
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
