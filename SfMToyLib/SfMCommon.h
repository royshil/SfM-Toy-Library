/*
 * SfMCommon.h
 *
 *  Created on: May 28, 2016
 *      Author: roy_shilkrot
 * 
 *  Copyright @  2016
 */

#ifndef SFMTOYLIB_SFMCOMMON_H_
#define SFMTOYLIB_SFMCOMMON_H_

#include <opencv2/core/core.hpp>

#include <map>

namespace sfmtoylib {

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
    /**
     * 3D point.
     */
    cv::Point3f p;

    /**
     * A mapping from image index to 2D point index in that image's list of features.
     */
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
 * Prune the features according to a binary mask (> 0).
 * @param features       features to prune
 * @param mask           mask to prune by
 * @param prunedFeatures pruned features
 */
void PruneFeaturesWithMask(const Features& features, const cv::Mat& mask, Features& prunedFeatures);

}  // namespace sfmtoylib


#endif /* SFMTOYLIB_SFMCOMMON_H_ */
