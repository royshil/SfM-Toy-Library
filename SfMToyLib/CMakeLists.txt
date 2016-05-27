cmake_minimum_required(VERSION 3.2)

set(SFM_LIB_SOURCES SfM.h                   SfM.cpp
                    SfM2DFeatureUtilities.h SfM2DFeatureUtilities.cpp
    )

add_library(SfMToyLibrary ${SFM_LIB_SOURCES} )