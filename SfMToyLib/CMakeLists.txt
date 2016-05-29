cmake_minimum_required(VERSION 3.2)

set(SFM_LIB_SOURCES SfMCommon.h             SfMCommon.cpp
                    SfM.h                   SfM.cpp
                    SfM2DFeatureUtilities.h SfM2DFeatureUtilities.cpp
                    SfMStereoUtilities.h    SfMStereoUtilities.cpp
    )

add_library(SfMToyLibrary ${SFM_LIB_SOURCES} )