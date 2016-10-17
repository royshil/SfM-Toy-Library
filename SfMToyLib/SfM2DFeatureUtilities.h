/*
 * SfM2DFeatureUtilities.h
 *
 *  Created on: May 26, 2016
 *      Author: roy_shilkrot
 * 
 *  Copyright @ Roy Shilkrot 2016
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 */

#ifndef SFMTOYLIB_SFM2DFEATUREUTILITIES_H_
#define SFMTOYLIB_SFM2DFEATUREUTILITIES_H_

#include "SfMCommon.h"

#include <opencv2/features2d.hpp>

namespace sfmtoylib {

class SfM2DFeatureUtilities {
public:
    SfM2DFeatureUtilities();
    virtual ~SfM2DFeatureUtilities();

    Features extractFeatures(const cv::Mat& image);

    static Matching matchFeatures(
            const Features& featuresLeft,
            const Features& featuresRight);

private:
    cv::Ptr<cv::Feature2D>         mDetector;
    cv::Ptr<cv::DescriptorMatcher> mMatcher;

};

} /* namespace sfmtoylib */

#endif /* SFMTOYLIB_SFM2DFEATUREUTILITIES_H_ */
