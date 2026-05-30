cmake_minimum_required(VERSION 3.20)
project(NeuralEngineVisualizer LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD 17)

find_package(OpenCV        REQUIRED)
find_package(PkgConfig     REQUIRED)
find_package(nlohmann_json REQUIRED)

# IMPORTED_TARGET = cmake handles include + lib path automatically
pkg_check_modules(ZMQ REQUIRED IMPORTED_TARGET libzmq)

add_executable(engine
    main.cpp
    inference.mm
)

target_include_directories(engine PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    /opt/homebrew/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(engine PRIVATE
    ${OpenCV_LIBS}
    PkgConfig::ZMQ
    nlohmann_json::nlohmann_json
    "-framework CoreML"
    "-framework CoreVideo"
    "-framework Foundation"
)

target_compile_options(engine PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -O2>
    $<$<COMPILE_LANGUAGE:OBJCXX>:-Wall -O2 -fobjc-arc>
)