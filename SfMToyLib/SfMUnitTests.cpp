/*
 * SfMUnitTests.cpp
 *
 *  Created on: Oct 6, 2016
 *      Author: roy_shilkrot
 *
 * Copyright @ Roy Shilkrot 2016.
 *
 * Unit tests for the SfM pipeline.
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
#include "SfMStereoUtilities.h"

#define BOOST_TEST_MODULE sfm_unit_tests
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#define BOOST_TEST_EQUALS(a, b, e) BOOST_TEST((((a) == (b)) || (std::abs((a) - (b)) < e)))

using namespace sfmtoylib;
using namespace std;

BOOST_AUTO_TEST_SUITE( SfMUnitTests )

//simple 640x480 camera intrinsics
const float FOCAL_FACTOR = 700.0f;
const cv::Matx33f INTRINSICS(FOCAL_FACTOR, 0, 320.0f,
                             0, FOCAL_FACTOR, 240.0f,
                             0,            0,   1.0f);

//some random 3D points
const Points3f cannedPoints3d {
    { 4, 12, 50},
    {12, 11, 55},
    {22,  1, 45},
    {13,  3, 60},
    {11, 16, 61},
    {21, 12, 65},
    {24, 11, 67},
    {29,  6, 41},
    {27,  4, 44},
    {22,  7, 58},
    {20,  9, 51},
    {15, 10, 40}};

/**
 * Generate aligned 2D-3D points using a mock camera and projection.
 * @param points3d         output: 3D points
 * @param imagePoints    output: 2D points that correspond to the 3D points
 * @param rotationMatrix output: camera rotation matrix (3x3)
 * @param translation    output: camera trasnlation vector (1x3)
 */
void generate2DPointsFromMockCamera(Points3f& points3d, Points2f& imagePoints, cv::Matx33f& rotationMatrix, cv::Vec3f& translation) {
    //create random camera matrix
    const cv::Vec3f eulerAngles(5, 5, 5);
    translation = cv::Vec3f(-10, 0, 30);

    ceres::EulerAnglesToRotationMatrix<float>(eulerAngles.val, 3, rotationMatrix.val);

    cv::Vec3f rotationVector;
    cv::Rodrigues(rotationMatrix, rotationVector);

    points3d = cannedPoints3d;

    //project using OpenCV
    imagePoints.resize(points3d.size());
    cv::projectPoints(points3d, rotationVector, translation, INTRINSICS, cv::Mat(), imagePoints);
}

/**
 * Generate 2D points in 2 cameras looking at the same "scene"
 * @param points3d    output: originating 3D points
 * @param leftImage   output: left image 2D points
 * @param rightImage  output: right image 2D points
 * @param leftCamera  output: left camera pose
 * @param rightCamera output: right camera pose
 */
void generateStereoViews(Points3f& points3d, Points2f& leftImage, Points2f& rightImage, cv::Matx34f& leftCamera, cv::Matx34f& rightCamera) {
    //create random camera matrices
    const cv::Vec3f leftEulerAngles(5, 5, 5);
    leftCamera(0, 3) = -10;
    leftCamera(1, 3) = 0;
    leftCamera(2, 3) = 30;

    const cv::Vec3f rightEulerAngles(-5, 0, 5);
    rightCamera(0, 3) = 10;
    rightCamera(1, 3) = 0;
    rightCamera(2, 3) = 28;

    points3d = cannedPoints3d;

    //------- Left camera
    cv::Matx33f rotationMatrix;
    ceres::EulerAnglesToRotationMatrix<float>(leftEulerAngles.val, 3, rotationMatrix.val);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            leftCamera(i, j) = rotationMatrix(i, j);
        }
    }

    cv::Vec3f rotationVector;
    cv::Rodrigues(rotationMatrix, rotationVector);

    leftImage.resize(points3d.size());
    cv::projectPoints(points3d, rotationVector, leftCamera.get_minor<3, 1>(0, 3), INTRINSICS, cv::Mat(), leftImage);

    //------- Right camera
    ceres::EulerAnglesToRotationMatrix<float>(rightEulerAngles.val, 3, rotationMatrix.val);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            rightCamera(i, j) = rotationMatrix(i, j);
        }
    }

    cv::Rodrigues(rotationMatrix, rotationVector);

    rightImage.resize(points3d.size());
    cv::projectPoints(points3d, rotationVector, rightCamera.get_minor<3, 1>(0, 3), INTRINSICS, cv::Mat(), rightImage);
}

/**
 * This unit test checks if the OpenCV reprojection of 3D points to 2D is similar to that
 * which happens in the Ceres reprojection error function for optimizaion. If it succeeds
 * we can be sure Ceres is optimizing over the right residuals.
 */
BOOST_AUTO_TEST_CASE( ceres_reprojection_test )
{
    Points3f points3d;
    Points2f points2d;
    cv::Matx33f rotationMatrix;
    cv::Vec3f translation;
    generate2DPointsFromMockCamera(points3d, points2d, rotationMatrix, translation);

    float angleAxis[3];
    ceres::RotationMatrixToAngleAxis<float>(rotationMatrix.t().val, angleAxis); //assumes col-major!

    //project using Ceres-manual and check vs. OpenCV
    for (size_t i = 0; i < points3d.size(); i++) {
        const cv::Point3f& p3d = points3d[i];

        float p[3];
        //rotate
        ceres::AngleAxisRotatePoint(angleAxis, &(p3d.x), p);

        //translate
        p[0] += translation(0);
        p[1] += translation(1);
        p[2] += translation(2);

        //perspective divide
        const float xp = p[0] / p[2];
        const float yp = p[1] / p[2];

        //focal and center of projection
        const cv::Point2f predicted(FOCAL_FACTOR * xp + INTRINSICS(0, 2),
                                    FOCAL_FACTOR * yp + INTRINSICS(1, 2));

        // Check Ceres-manual projection vs OpenCV projection
        BOOST_TEST_EQUALS(predicted.x, points2d[i].x, 0.1);
        BOOST_TEST_EQUALS(predicted.y, points2d[i].y, 0.1);
    }
}

/**
 * Test the correctness of `SfMStereoUtilities::findCameraPoseFrom2D3DMatch`
 */
BOOST_AUTO_TEST_CASE(find_camera_pose_from_2d3d_match) {
    //generate mock data: a 2D view of the sparse 3D scene
    Points3f points3d;
    Points2f points2d;
    cv::Matx33f rotationMatrix;
    cv::Vec3f translation;
    generate2DPointsFromMockCamera(points3d, points2d, rotationMatrix, translation);

    //recover the pose from 2D-3D correspondence
    Pose recoveredPose;
    SfMStereoUtilities::findCameraPoseFrom2D3DMatch(
            { cv::Mat(INTRINSICS), cv::Mat(INTRINSICS.inv()), cv::Mat()}, //intrinsics
            { points2d, points3d },                                          //aligned points
            recoveredPose);

    //test the recovered pose vs. the ground-truth pose
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            BOOST_TEST_EQUALS(rotationMatrix(i, j), recoveredPose(i, j), 0.01);
        }
        BOOST_TEST_EQUALS(translation(i), recoveredPose(i, 3), 0.1);
    }
}

/**
 * Test the correctness of `SfMStereoUtilities::triangulateViews`.
 */
BOOST_AUTO_TEST_CASE(triangulate_from_2_views) {
    //Generate mock data: 2 views of sparse 3D scene
    Points3f points3d;
    Points2f leftImage;
    Points2f rightImage;
    cv::Matx34f leftCamera;
    cv::Matx34f rightCamera;
    generateStereoViews(points3d, leftImage, rightImage, leftCamera, rightCamera);

    Matching matching = GetAlignedMatching(leftImage.size());

    PointCloud pointCloud;

    //triangulate
    SfMStereoUtilities::triangulateViews(
            { cv::Mat(INTRINSICS), cv::Mat(INTRINSICS.inv()), cv::Mat()}, //intrinsics
            { 0, 1 },                                                     //image pair
            matching,                                                     //aligned matching
            { PointsToKeyPoints(leftImage), leftImage, cv::Mat::zeros(leftImage.size(), 1, CV_32FC1) },    //left features
            { PointsToKeyPoints(rightImage), rightImage, cv::Mat::zeros(rightImage.size(), 1, CV_32FC1) }, //right features
            leftCamera,                                                   //left camera pose
            rightCamera,                                                  //right camera pose
            pointCloud                                                    //output pointcloud
            );

    //test triangulation result
    for (size_t i = 0; i < pointCloud.size(); i++) {
        const Point3DInMap& p = pointCloud[i];
        BOOST_TEST_EQUALS(norm(p.p - points3d[i]), 0.0, 0.01);
    }
}

BOOST_AUTO_TEST_SUITE_END()
