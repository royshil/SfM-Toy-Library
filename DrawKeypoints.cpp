#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <libgen.h>

using namespace cv;

int main( int argc, char** argv )
{
	Mat img_1 = imread( argv[1] );
		
	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;
	
	SurfFeatureDetector detector( minHessian );
	
	std::vector<KeyPoint> keypoints_1;
	
	detector.detect( img_1, keypoints_1 );
	
	//-- Draw keypoints
	Mat img_keypoints_1;
	
	drawKeypoints( img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
	
	//-- Show detected (drawn) keypoints
//	imshow("Keypoints 1", img_keypoints_1 );
	
	imwrite(std::string(basename(argv[1])) + "_keypoints.jpg", img_keypoints_1);
	
	return 0;
}
