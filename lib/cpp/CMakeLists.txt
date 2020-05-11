CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
PROJECT(CppToolKit CXX Fortran)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native -Wall -Wextra -lgfortran -lpthread -fPIC -fopenmp")
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Ofast -march=native -fPIC")

ADD_LIBRARY(FortranDeps SHARED MathUtils.f90)
#ADD_EXECUTABLE(CppToolKit cpp_code.cpp)
ADD_LIBRARY(CppToolKit SHARED cpp_code.cpp)


TARGET_LINK_LIBRARIES(CppToolKit FortranDeps)
TARGET_COMPILE_OPTIONS(CppToolKit PRIVATE -std=c++17)
