CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
PROJECT(CppIntegrand)

FIND_LIBRARY(SPLINTER_LIBRARY NAMES splinter-3-0 PATHS /home/amaraio/Documents/Splinter/SplinterBuild/lib REQUIRED)   
INCLUDE_DIRECTORIES(/home/amaraio/Documents/Splinter/SplinterBuild/include/SPLINTER)

FIND_LIBRARY(CUBA_LIBRARY NAMES cuba PATHS /home/amaraio/Documents/Cuba/cuba REQUIRED)   
INCLUDE_DIRECTORIES(/home/amaraio/Documents/Cuba/cuba)

SET(CMAKE_CXX_FLAGS_RELEASE "-O2  -shared -fPIC  -DNDEBUG ")
SET(CMAKE_C_FLAGS_RELEASE "-O2  -DNDEBUG ")

ADD_LIBRARY(CppIntegrand SHARED CppIntegrand.cpp)
TARGET_LINK_LIBRARIES(CppIntegrand ${SPLINTER_LIBRARY} ${CUBA_LIBRARY})
TARGET_COMPILE_OPTIONS(CppIntegrand PRIVATE -std=c++14 -shared -fPIC -O2 -Wno-ignored-attributes -Wno-deprecated-declarations)
