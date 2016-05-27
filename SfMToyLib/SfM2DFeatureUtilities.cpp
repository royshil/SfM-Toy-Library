/*
 * SfM2DFeatureUtilities.cpp
 *
 *  Created on: May 26, 2016
 *      Author: roy_shilkrot
 * 
 *  Copyright @  2016
 */

#include "SfM2DFeatureUtilities.h"

namespace sfmtoylib {

SfM2DFeatureUtilities::SfM2DFeatureUtilities() {
    // TODO Auto-generated constructor stub

}

SfM2DFeatureUtilities::~SfM2DFeatureUtilities() {
    // TODO Auto-generated destructor stub
}

void SfM2DFeatureUtilities::extractFeatures(const cv::Mat& image, FeaturePointsDescriptors& features) {
}

void SfM2DFeatureUtilities::matchFeatures(const FeaturePointsDescriptors& featuresLeft,
        const FeaturePointsDescriptors& featuresRight, std::vector<cv::DMatch>& matching) {
}

} /* namespace sfmtoylib */
