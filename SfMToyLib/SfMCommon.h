/*
 * SfMCommon.h
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

#ifndef SFMTOYLIB_SFMCOMMON_H_
#define SFMTOYLIB_SFMCOMMON_H_

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <map>

namespace sfmtoylib {

enum DebugLogLevel {
    LOG_TRACE = 0,
    LOG_DEBUG,
    LOG_INFO,
    LOG_WARN,
    LOG_ERROR
};

///Rotational element in a 3x4 matrix
const cv::Rect ROT(0, 0, 3, 3);

///Translational element in a 3x4 matrix
const cv::Rect TRA(3, 0, 1, 3);

///Minimal ratio of inliers-to-total number of points for computing camera pose
const float POSE_INLIERS_MINIMAL_RATIO = 0.5;

struct Intrinsics {
    cv::Mat K;
    cv::Mat Kinv;
    cv::Mat distortion;
};

struct ImagePair {
    size_t left, right;
};

std::ostream& operator<< (std::ostream& stream, const ImagePair& pair);

typedef std::vector<cv::KeyPoint> Keypoints;
typedef std::vector<cv::Point2f>  Points2f;
typedef std::vector<cv::Point3f>  Points3f;

struct Image2D3DMatch {
    Points2f points2D;
    Points3f points3D;
};

struct Features {
    Keypoints keyPoints;
    Points2f  points;
    cv::Mat   descriptors;
};

struct Point3DInMap {
    // 3D point.
    cv::Point3f p;

    // A mapping from image index to 2D point index in that image's list of features.
    std::map<int, int> originatingViews;
};

struct Point3DInMapRGB {
    Point3DInMap p;
    cv::Scalar   rgb;
};

typedef std::vector<cv::DMatch>      Matching;
typedef std::vector<Point3DInMap>    PointCloud;
typedef std::vector<Point3DInMapRGB> PointCloudRGB;

typedef cv::Matx34f Pose;

/**
 * Get the features for left and right images after keeping only the matched features and aligning them.
 * Alignment: i-th feature in left is a match to i-th feature in right.
 * @param leftFeatures       Left image features.
 * @param rightFeatures      Right image features.
 * @param matches            Matching over the features.
 * @param alignedLeft        Output: aligned left features.
 * @param alignedRight       Output: aligned right features.
 * @param leftBackReference  Output: back reference from aligned index to original index
 * @param rightBackReference Output: back reference from aligned index to original index
 */
void GetAlignedPointsFromMatch(const Features& leftFeatures,
                               const Features& rightFeatures,
                               const Matching& matches,
                               Features& alignedLeft,
                               Features& alignedRight,
                               std::vector<int>& leftBackReference,
                               std::vector<int>& rightBackReference);

/**
 * Get the features for left and right images after keeping only the matched features and aligning them.
 * Alignment: i-th feature in left is a match to i-th feature in right.
 * @param leftFeatures  Left image features.
 * @param rightFeatures Right image features.
 * @param matches       Matching over the features.
 * @param alignedLeft   Output: aligned left features.
 * @param alignedRight  Output: aligned right features.
 */
void GetAlignedPointsFromMatch(const Features& leftFeatures,
                               const Features& rightFeatures,
                               const Matching& matches,
                               Features& alignedLeft,
                               Features& alignedRight);

/**
 * Get a Matching for an aligned set: i -> i
 * @param size size of maching vector
 * @return aligned matching.
 */
Matching GetAlignedMatching(size_t size);

/**
 * Convert Keypoints to Points2f
 * @param kps keypoints
 * @param ps  points
 */
void KeyPointsToPoints(const Keypoints& kps, Points2f& ps);

/**
 * Convert Points2f to Keypoints.
 * Note: distance on Keypoint will be set to 1.0.
 * @param ps  Points
 * @param kps Keypoints
 */
void PointsToKeyPoints(const Points2f& ps, Keypoints& kps);

/**
 * Convert Points2f to Keypoints.
 * Note: distance on Keypoint will be set to 1.0.
 * @param ps points
 * @return keypoints
 */
Keypoints PointsToKeyPoints(const Points2f& ps);

/**
 * Prune the features according to a binary mask (> 0).
 * @param features       features to prune
 * @param mask           mask to prune by
 * @param prunedFeatures pruned features
 */
void PruneFeaturesWithMask(const Features& features, const cv::Mat& mask, Features& prunedFeatures);

/**
 * `cv::imshow` version with image scaling (`cv::resize`)
 * @param windowName window name
 * @param image		 image to show
 * @param scale		 scale to use in `cv::resize`
 */
void imshow(const std::string& windowName, const cv::Mat& image, const double scale);

namespace Colors {
    const cv::Scalar BLUE   = cv::Scalar(255,   0,   0);
    const cv::Scalar AQUA   = cv::Scalar(255, 128,   0);
    const cv::Scalar CYAN   = cv::Scalar(255, 255,   0);
    const cv::Scalar MARINE = cv::Scalar(128, 255,   0);
    const cv::Scalar GREEN  = cv::Scalar(  0, 255,   0);
    const cv::Scalar LIME   = cv::Scalar(  0, 255, 128);
    const cv::Scalar YELLOW = cv::Scalar(  0, 255, 255);
    const cv::Scalar ORANGE = cv::Scalar(  0, 128, 255);
    const cv::Scalar RED    = cv::Scalar(  0,   0, 255);
    const cv::Scalar BEIGE  = cv::Scalar(128,   0, 255);
    const cv::Scalar PURPLE = cv::Scalar(255,   0, 255);
    const cv::Scalar DEEP   = cv::Scalar(255,   0, 128);
    const cv::Scalar WHITE  = cv::Scalar::all(255);
    const cv::Scalar BLACK  = cv::Scalar::all(0);

    const std::vector<cv::Scalar> WHEEL = {
            BLUE  ,
            AQUA  ,
            CYAN  ,
            MARINE,
            GREEN ,
            LIME  ,
            YELLOW,
            ORANGE,
            RED   ,
            BEIGE ,
            PURPLE,
            DEEP  ,
            BLACK
    };
}

}  // namespace sfmtoylib


#endif /* SFMTOYLIB_SFMCOMMON_H_ */
