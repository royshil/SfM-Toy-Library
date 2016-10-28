#  The MIT License (MIT)
# 
#  Copyright (c) 2013 Roy Shilkrot
#   
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#   
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#   
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

project(SfMToyExample)
cmake_minimum_required(VERSION 3.2)

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(OpenCV 3.1 REQUIRED COMPONENTS core calib3d features2d imgproc highgui)
find_package(Boost      REQUIRED COMPONENTS system chrono filesystem unit_test_framework program_options)
find_package(Ceres      REQUIRED)
find_package(OpenGL)

# enable C++11 standard
set(CXX_STANDARD 11)
set(CXX_STANDARD_REQUIRED ON)

include_directories(${Boost_INCLUDE_DIR})
link_directories   (${Boost_LIBRARY_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})

add_subdirectory(SfMToyLib)

IF(APPLE)
	set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -F/usr/local/lib -L/opt/local/lib")
	INCLUDE_DIRECTORIES ( /System/Library/Frameworks )
ENDIF(APPLE)

set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D__SFM__DEBUG__" )

include_directories(SfMToyLib)

# Command-line app
add_executable(SfMToyUI main.cpp)

set(SFM_TOY_APP_LIBS    
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
        ${CERES_LIBRARIES}
    )

target_link_libraries(SfMToyUI 
	SfMToyLibrary
	${SFM_TOY_APP_LIBS}
	)

