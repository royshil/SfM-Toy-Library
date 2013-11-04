project(SfMToyExample)
cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)
find_package(OpenMP)
find_package(OpenGL)

add_subdirectory(SfMToyLib)

#find_package(PCL 1.5)

############### Find Eigen ###############
set(EIGEN_DIR_HINT "" CACHE PATH "Eigen3 directory")
find_path(EIGEN_INCLUDE "Eigen/Eigen" PATHS ${EIGEN_DIR_HINT})
if(NOT EIGEN_INCLUDE)
	message(FATAL_ERROR "Eigen3 includes not found")
ENDIF(NOT EIGEN_INCLUDE)
include_directories(${EIGEN_INCLUDE})
###########################################


SET(USE_GUI TRUE CACHE BOOL "Build with GUI or just CLI")
IF(USE_GUI)
	ADD_DEFINITIONS("-DHAVE_GUI")
ENDIF(USE_GUI)

SET(USE_QT TRUE CACHE BOOL "Use QT for GUI")
IF(USE_QT)
	########## Qt stuff ##########
	find_package(Qt4 REQUIRED QtGui QtOpenGL QtXml)
	
	ADD_DEFINITIONS(${QT_DEFINITIONS})
	include_directories(${QT_INCLUDES})
	include(${QT_USE_FILE}) 
	
	QT4_WRAP_UI( MY_UI_HDRS sfmtoy.ui )
	QT4_WRAP_CPP( MY_MOC_SRCS ViewerInterface.h sfmviewer.h )
	SOURCE_GROUP(QT_MOC_SRCS FILES ${MY_MOC_SRCS})
	
	INCLUDE_DIRECTORIES( ${CMAKE_BINARY_DIR} ) #qt wrapping creates some sources
	######### End Qt Stuff #########
	
	############### Find LibQGLViewer ###############
	set(QGLVIEWER_DIR_HINT "" CACHE PATH "libQGLViewer directory")
	FIND_LIBRARY(QGLVIEWER_LIB QGLViewer2 PATHS ${QGLVIEWER_DIR_HINT} "${QGLVIEWER_DIR_HINT}/QGLViewer/release")
	IF(NOT QGLVIEWER_LIB)
	    FIND_LIBRARY(QGLVIEWER_LIB QGLViewer PATHS ${QGLVIEWER_DIR_HINT} "${QGLVIEWER_DIR_HINT}/QGLViewer/release")
	ENDIF()
	IF(NOT QGLVIEWER_LIB)
	   message(FATAL_ERROR "QGLVIEWER not found")
	ENDIF(NOT QGLVIEWER_LIB)
	find_path(QGLVIEWER_INCLUDE "QGLViewer/qglviewer.h" PATHS ${QGLVIEWER_DIR_HINT})
	if(NOT QGLVIEWER_INCLUDE)
		message(FATAL_ERROR "QGLViewer includes not found")
	ENDIF(NOT QGLVIEWER_INCLUDE)
	include_directories(${QGLVIEWER_INCLUDE})
	############################################
ENDIF(USE_QT)


IF(APPLE)
#	set( COCOA_LIBS ${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/Cocoa.framework )
	set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -F/usr/local/lib -L/opt/local/lib")
	INCLUDE_DIRECTORIES ( /System/Library/Frameworks )
	FIND_LIBRARY(COCOA_LIBRARY Cocoa)
	FIND_LIBRARY(GLUT_LIBRARY GLUT )
	SET(OpenGL_LIBS ${COCOA_LIBRARY} ${GLUT_LIBRARY} ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY})
ENDIF(APPLE)

#if(PCL_FOUND)
#	link_directories(${PCL_LIBRARY_DIRS})
#	
#	add_definitions(${PCL_DEFINITIONS})
#
#	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DHAVE_PCL")
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_PCL")
#endif()

set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D__SFM__DEBUG__" )

if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -DHAVE_OPENMP")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -DHAVE_OPENMP")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(
#	${FLTK_INCLUDE_DIR} 
#	${PCL_INCLUDE_DIRS} 
	SfMToyLib
	3rdparty/SSBA-3.0
	)

# GUI part
add_executable(SfMToyUI 
	ViewerInterface.h ViewerInterface.cpp
	sfmviewer.h sfmviewer.cpp
#	Visualization.h Visualization.cpp
	${MY_UI_HDRS}
	${MY_MOC_SRCS}
	main.cpp
	)

target_link_libraries(SfMToyUI 
	SfMToyLibrary
	${OpenCV_LIBS}
#	${PCL_LIBRARIES}
	${QT_LIBRARIES} 
	${QGLVIEWER_LIB}
	${OpenGL_LIBS}
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

