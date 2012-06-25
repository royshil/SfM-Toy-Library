project(SfMToyExample)
cmake_minimum_required(VERSION 2.8)

find_package(PCL REQUIRED)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

############ Find FLTK ############
if(NOT DEFINED FLTK_PATH)
    file(GLOB_RECURSE FLTK_PATH "${CMAKE_SOURCE_DIR}/Widget.h")
    if(FLTK_PATH STREQUAL FLTK_PATH-NOTFOUND   OR   FLTK_PATH STREQUAL "")
		set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNO_FLTK")
		set( FLTK-NOTFOUND TRUE )
        message(STATUS "FLTK was not found !!!!!")
    else()
        list(LENGTH FLTK_PATH FLTK_PATH_LEN)
        if(FLTK_PATH_LEN GREATER 1)
            list(GET FLTK_PATH 1 FLTK_PATH)
        endif()
        file(RELATIVE_PATH FLTK_PATH ${CMAKE_SOURCE_DIR} ${FLTK_PATH})
        get_filename_component(FLTK_PATH ${FLTK_PATH} REALPATH)
        get_filename_component(FLTK_PATH ${FLTK_PATH} PATH)
        message(STATUS "Found FLTK at ${FLTK_PATH}")
    endif()
else()
    get_filename_component(FLTK_PATH ${FLTK_PATH} REALPATH)
    message(STATUS "FLTK path set to ${FLTK_PATH}")
endif()
set(FLTK_INCLUDE_DIR ${FLTK_PATH}/include)
set(FLTK_LIB_DIR ${FLTK_PATH}/lib)
set(FLTK_LIBRARIES fltk3 fltk3gl)
##################################

IF(APPLE)
	set( COCOA_LIBS ${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/Cocoa.framework )
	set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -F/usr/local/lib -L/opt/local/lib")
ENDIF(APPLE)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS} ${FLTK_LIB_DIR})
add_definitions(${PCL_DEFINITIONS})
include_directories(${FLTK_INCLUDE_DIR})

set(SSBA_LIBRARY_DIR SSBA_LIBRARY_DIR-NOT_FOUND CACHE PATH "Directory to find SSBA libs")

link_directories(${SSBA_LIBRARY_DIR})
include_directories(${SSBA_LIBRARY_DIR}/../)

set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D__SFM__DEBUG__" )

# Core functions of SfM
file(GLOB SFM_LIB_SOURCES	
	Distance.* 
	Triangulation.* 
	FeatureMatching.*
	FindCameraMatrices.*
	MultiCameraDistance.*
	IDistance.*
	MultiCameraPnP.*
	Common.*
	IFeatureMatcher.*
	RichFeatureMatcher.*
	OFFeatureMatcher.*
	BundleAdjuster.*
	GPUSURFFeatureMatcher.*
)

add_library(SfMToyLibrary ${SFM_LIB_SOURCES} )
target_link_libraries(SfMToyLibrary 
	V3D 
	COLAMD)

# GUI part
IF( NOT ${FLTK-NOTFOUND} )
	message(STATUS "Adding fltk to GUI")
	add_executable(SfMToyUI 
		DistanceUI.cpp 
		DistanceUI.h 
		Visualization.cpp
		Visualization.h
		gui.cxx
		gui.h
		)
	
	target_link_libraries(SfMToyUI 
		${FLTK_LIBRARIES})
ELSE()
	message(STATUS "not using fltk in GUI")
	add_executable(SfMToyUI 
		Visualization.cpp
		Visualization.h
		main.cpp
		)
endif()

target_link_libraries(SfMToyUI 
	SfMToyLibrary
	${OpenCV_LIBS}
	${PCL_LIBRARIES}
	)

IF(APPLE)
	target_link_libraries(SfMToyUI ${COCOA_LIBS})
ENDIF(APPLE)


add_executable(TestStuff
	DrawKeypoints.cpp
	)
target_link_libraries(TestStuff
	SfMToyLibrary
	${OpenCV_LIBS}
	)

add_executable(TestOpenCVRotations
	rotations.cpp
	)
target_link_libraries(TestOpenCVRotations
	${OpenCV_LIBS}
	)
