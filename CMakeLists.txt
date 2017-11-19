cmake_minimum_required(VERSION 3.10.0)

project(Vanilla)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY build)


set(SOURCE_FILES src/main.cpp header/pretty_print.hpp header/init.hpp header/activation_functions.hpp header/model.hpp src/model.cpp)
find_package(Boost 1.65.1 COMPONENTS)

if(Boost_FOUND)
        # MESSAGE ("-- Boost Found")
	include_directories(${Boost_INCLUDE_DIRS}) 
    	add_executable(Vanilla ${SOURCE_FILES})
	target_link_libraries(Vanilla ${Boost_LIBRARIES})
endif()
