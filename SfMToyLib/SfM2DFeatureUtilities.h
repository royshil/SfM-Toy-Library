/*
 * SfM2DFeatureUtilities.h
 *
 *  Created on: May 26, 2016
 *      Author: roy_shilkrot
 * 
 *  Copyright @ Roy Shilkrot 2016
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
