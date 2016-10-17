/*
 * SfM2DFeatureUtilities.cpp
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

#include "SfM2DFeatureUtilities.h"

using namespace cv;
using namespace std;

namespace sfmtoylib {

const double NN_MATCH_RATIO = 0.8f; // Nearest-neighbour matching ratio

SfM2DFeatureUtilities::SfM2DFeatureUtilities() {
    // initialize detector and extractor
    mDetector = ORB::create(5000);
    mMatcher = DescriptorMatcher::create("BruteForce-Hamming");
}

SfM2DFeatureUtilities::~SfM2DFeatureUtilities() {
}

Features SfM2DFeatureUtilities::extractFeatures(const cv::Mat& image) {
    Features features;
    mDetector->detectAndCompute(image, noArray(), features.keyPoints, features.descriptors);
    KeyPointsToPoints(features.keyPoints, features.points);
    return features;
}

Matching SfM2DFeatureUtilities::matchFeatures(
        const Features& featuresLeft,
        const Features& featuresRight) {
    //initial matching between features
    vector<Matching> initialMatching;

    auto matcher = DescriptorMatcher::create("BruteForce-Hamming");
    matcher->knnMatch(featuresLeft.descriptors, featuresRight.descriptors, initialMatching, 2);

    //prune the matching using the ratio test
    Matching prunedMatching;
    for(unsigned i = 0; i < initialMatching.size(); i++) {
        if(initialMatching[i][0].distance < NN_MATCH_RATIO * initialMatching[i][1].distance) {
            prunedMatching.push_back(initialMatching[i][0]);
        }
    }

    return prunedMatching;
}

} /* namespace sfmtoylib */
