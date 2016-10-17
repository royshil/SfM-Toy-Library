#  The MIT License (MIT)
# 
#  Copyright (c) 2016 Roy Shilkrot
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