/*
 * SfMCommon.cpp
 *
 *  Created on: May 28, 2016
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

#include "SfMCommon.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <ostream>

using namespace std;
using namespace cv;

namespace sfmtoylib {

std::ostream& operator<< (std::ostream& stream, const ImagePair& pair) {
    return stream << "[" << pair.left << ", " << pair.right << "]";
}

void GetAlignedPointsFromMatch(const Features& leftFeatures,
                               const Features& rightFeatures,
                               const Matching& matches,
                               Features& alignedLeft,
                               Features& alignedRight)
{
    vector<int> leftBackReference, rightBackReference;
    GetAlignedPointsFromMatch(
            leftFeatures,
            rightFeatures,
            matches,
            alignedLeft,
            alignedRight,
            leftBackReference,
            rightBackReference
            );

}

void GetAlignedPointsFromMatch(const Features& leftFeatures,
                               const Features& rightFeatures,
                               const Matching& matches,
                               Features& alignedLeft,
                               Features& alignedRight,
                               vector<int>& leftBackReference,
                               vector<int>& rightBackReference)
{
    alignedLeft .keyPoints.clear();
    alignedRight.keyPoints.clear();
    alignedLeft .descriptors = cv::Mat();
    alignedRight.descriptors = cv::Mat();

    for (unsigned int i=0; i<matches.size(); i++) {
        alignedLeft .keyPoints  .push_back(leftFeatures.keyPoints       [matches[i].queryIdx]);
        alignedLeft .descriptors.push_back(leftFeatures.descriptors.row (matches[i].queryIdx));
        alignedRight.keyPoints  .push_back(rightFeatures.keyPoints      [matches[i].trainIdx]);
        alignedRight.descriptors.push_back(rightFeatures.descriptors.row(matches[i].trainIdx));
        leftBackReference .push_back(matches[i].queryIdx);
        rightBackReference.push_back(matches[i].trainIdx);
    }

    KeyPointsToPoints(alignedLeft.keyPoints,  alignedLeft.points);
    KeyPointsToPoints(alignedRight.keyPoints, alignedRight.points);
}

void KeyPointsToPoints(const Keypoints& kps, Points2f& ps) {
    ps.clear();
    for (const auto& kp : kps) {
        ps.push_back(kp.pt);
    }
}

Keypoints PointsToKeyPoints(const Points2f& ps) {
	Keypoints kps;
	PointsToKeyPoints(ps, kps);
	return kps;
}

void PointsToKeyPoints(const Points2f& ps, Keypoints& kps) {
    kps.clear();
    for (const auto& p : ps) {
        kps.push_back(KeyPoint(p, 1.0f));
    }
}

Matching GetAlignedMatching(size_t size) {
    Matching match;
    for (size_t i = 0; i < size; i++) {
        match.push_back(DMatch(i, i, 0));
    }
    return match;
}

void PruneFeaturesWithMask(const Features& features, const cv::Mat& mask, Features& prunedFeatures) {
    prunedFeatures.keyPoints.clear();
    prunedFeatures.points.clear();
    prunedFeatures.descriptors = Mat();

    for (size_t i = 0; i < features.keyPoints.size(); i++) {
        if (mask.at<uchar>(i) > 0) {
            prunedFeatures.keyPoints.push_back(features.keyPoints[i]);
            prunedFeatures.points.push_back(features.points[i]);
            prunedFeatures.descriptors.push_back(features.descriptors.row(i));
        }
    }
}

void imshow(const std::string& windowName, const cv::Mat& image, const double scale) {
	Mat tmp;
	cv::resize(image, tmp, Size(), scale, scale);
	cv::imshow(windowName, tmp);
}

}  // namespace sfmtoylib

