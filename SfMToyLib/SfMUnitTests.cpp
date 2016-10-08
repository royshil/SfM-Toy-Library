/*
 * SfMUnitTests.cpp
 *
 *  Created on: Oct 6, 2016
 *      Author: roy_shilkrot
 *
 * Unit tests for the SfM pipeline.
 *
 */

#define BOOST_TEST_MODULE sfm_unit_tests
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

using namespace std;

#define BOOST_TEST_EQUALS(a, b, e) BOOST_TEST((((a) == (b)) || (fabsf((a) - (b)) < e)))

BOOST_AUTO_TEST_SUITE( SfMUnitTests )

/**
 * This unit test checks if the OpenCV reprojection of 3D points to 2D is similar to that
 * which happens in the Ceres reprojection error function for optimizaion. If it succeeds
 * we can be sure Ceres is optimizing over the right residuals.
 */
BOOST_AUTO_TEST_CASE( ceres_reprojection_test )
{
	//simple 640x480 camera intrinsics
	const float focal = 700.0f;
    const cv::Matx33f K(focal, 0, 320.0f,
    					0, focal, 240.0f,
						0,	   0,   1.0f);

	//create random camera matrix
	const cv::Vec3f eulerAngles(5, 5, 5);
	const cv::Vec3f translation(-10, 0, 30);

	cv::Matx33f rotationMatrix;
	ceres::EulerAnglesToRotationMatrix<float>(eulerAngles.val, 3, rotationMatrix.val);

	cv::Vec3f rotationVector;
	cv::Rodrigues(rotationMatrix, rotationVector);

	//some random 3D points
	const vector<cv::Point3f> points3d {
		{ 4, 12, 50},
		{12, 11, 55},
		{13,  3, 60},
		{15, 10, 40}};

	cout << cv::Mat(points3d) << endl;
	cout << rotationMatrix << endl;
	cout << rotationVector << endl;

	//project using OpenCV
	vector<cv::Point2f> imagePoints(points3d.size());
	cv::projectPoints(points3d, rotationVector, translation, K, cv::Mat(), imagePoints);
	cout << imagePoints << endl;

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

		const float xp = p[0] / p[2];
		const float yp = p[1] / p[2];

		const cv::Point2f predicted(focal * xp + K(0, 2),
									focal * yp + K(1, 2));

		cout << predicted << endl;

		// Check Ceres-manual projection vs OpenCV projection
		BOOST_TEST_EQUALS(predicted.x, imagePoints[i].x, 0.1);
		BOOST_TEST_EQUALS(predicted.y, imagePoints[i].y, 0.1);
    }
}

BOOST_AUTO_TEST_SUITE_END()
