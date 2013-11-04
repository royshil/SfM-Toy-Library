#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/nonfree/features2d.hpp>
#include "FeatureMatching.h"
#include "FindCameraMatrices.h"

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
	if (argc <= 1) {
		cout << "USAGE: " << argv[0] << " <image1> [image2]" << endl;
		return 0;
	}
	
	Mat img_1 = imread( argv[1] );
		
	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;
	
	SurfFeatureDetector detector( minHessian );
	std::vector<KeyPoint> keypoints_1;
	detector.detect( img_1, keypoints_1 );

	Mat img_keypoints;

	if (argc == 2) {
		
		drawKeypoints( img_1, keypoints_1, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
		
	} else if (argc == 3) {
		Mat img_2 = imread(argv[2]);
		
		vector<KeyPoint> keypoints_2;
		detector.detect( img_2, keypoints_2 );

		SurfDescriptorExtractor extractor(48, 12, true);
		Mat descriptors_1,descriptors_2;
		extractor.compute(img_1, keypoints_1, descriptors_1);
		extractor.compute(img_2, keypoints_2, descriptors_2);
		
		BFMatcher matcher(NORM_L2);
		vector<DMatch> Fmatches;
		matcher.match(descriptors_1, descriptors_2, Fmatches);
		
		//vector<Point2f> pts1,pts2;
//		for (unsigned int i=0; i<matches.size(); i++) {
//			pts1.push_back(keypoints_1[matches[i].queryIdx].pt);
//			pts2.push_back(keypoints_2[matches[i].trainIdx].pt);
//		}
//		vector<uchar> status;
//		Mat F = findFundamentalMat(pts1, pts2, FM_RANSAC, 0.5, 0.99, status);
//
//		cout << "F " << F << endl;
//
		vector<KeyPoint> kpts1,kpts2;
//		vector<DMatch> Fmatches;
//		for (unsigned int i=0; i<pts1.size(); i++) {
//			if (status[i]) {
//				cout << "Fmatch " << i << endl;
//				Fmatches.push_back(DMatch(kpts1.size(),kpts1.size(),1));
//				kpts1.push_back(KeyPoint(pts1[i],1));
//				kpts2.push_back(KeyPoint(pts2[i],1));
//			}
//		}
		GetFundamentalMat(keypoints_1, keypoints_2, kpts1, kpts2, Fmatches
#ifdef __SFM__DEBUG__
						  , img_1, img_2
#endif
						  );
		
		drawMatches(img_1, keypoints_1, img_2, keypoints_2, Fmatches, img_keypoints, Scalar::all(-1), Scalar::all(-1), Mat(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	}

	
	
	imwrite(std::string(argv[1]) + "_keypoints.png", img_keypoints);
	
	return 0;
}
