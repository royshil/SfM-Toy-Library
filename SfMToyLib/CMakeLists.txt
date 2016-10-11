cmake_minimum_required(VERSION 3.2)

####### SfM Library ########

set(SFM_LIB_SOURCES SfMCommon.h                SfMCommon.cpp
                    SfM.h                      SfM.cpp
                    SfM2DFeatureUtilities.h    SfM2DFeatureUtilities.cpp
                    SfMStereoUtilities.h       SfMStereoUtilities.cpp
                    SfMBundleAdjustmentUtils.h SfMBundleAdjustmentUtils.cpp
    )

add_library(SfMToyLibrary ${SFM_LIB_SOURCES} )



####### Unit Tests ########

add_executable(SfMUnitTests SfMUnitTests.cpp)

# setup Boost unit test framework
target_include_directories(SfMUnitTests PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(SfMUnitTests SfMToyLibrary ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${CERES_LIBRARIES} ${OpenCV_LIBS})

enable_testing()

add_test( NAME sfmtest COMMAND SfMUnitTests )