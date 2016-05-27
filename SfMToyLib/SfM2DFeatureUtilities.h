/*
 * SfM2DFeatureUtilities.h
 *
 *  Created on: May 26, 2016
 *      Author: roy_shilkrot
 * 
 *  Copyright @  2016
 */

#ifndef SFMTOYLIB_SFM2DFEATUREUTILITIES_H_
#define SFMTOYLIB_SFM2DFEATUREUTILITIES_H_

#include <opencv2/core/core.hpp>

namespace sfmtoylib {

struct FeaturePointsDescriptors {
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat                   descriptors;
};

class SfM2DFeatureUtilities {
public:
    SfM2DFeatureUtilities();
    virtual ~SfM2DFeatureUtilities();

    void extractFeatures(
            const cv::Mat&             image,
            FeaturePointsDescriptors&  features);

    void matchFeatures(
            const FeaturePointsDescriptors& featuresLeft,
            const FeaturePointsDescriptors& featuresRight,
            std::vector<cv::DMatch>&        matching);

};

} /* namespace sfmtoylib */

#endif /* SFMTOYLIB_SFM2DFEATUREUTILITIES_H_ */
