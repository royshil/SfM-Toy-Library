project(SfMToyExample)
cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)
find_package(OpenMP)

add_subdirectory(SfMToyLib)

find_package(PCL 1.5)

IF(APPLE)
	set( COCOA_LIBS ${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/Cocoa.framework )
	set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -F/usr/local/lib -L/opt/local/lib -fopenmp")
ENDIF(APPLE)

if(PCL_FOUND)
	link_directories(${PCL_LIBRARY_DIRS})
	
	add_definitions(${PCL_DEFINITIONS})

	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DHAVE_PCL")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_PCL")
endif()

set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D__SFM__DEBUG__" )

if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -DHAVE_OPENMP")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -DHAVE_OPENMP")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(
	${FLTK_INCLUDE_DIR} 
	${PCL_INCLUDE_DIRS} 
	SfMToyLib
	)

# GUI part
add_executable(SfMToyUI 
	Visualization.cpp
	Visualization.h
	main.cpp
	)

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

