cmake_minimum_required(VERSION 3.10)
project(RandomForest)

set(CMAKE_CXX_FLAGS "-O2")

set(CMAKE_CXX_STANDARD 11)

add_executable(RandomForest main.cpp include/Data.h src/Data.cpp
        src/DecisionTree.cpp include/DecisionTree.h src/DecisionTree.cpp
        include/RandomForest.h src/RandomForest.cpp)
