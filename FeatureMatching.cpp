/*
 *  FeatureMatching.cpp
 *  EyeRingOpenCV
 *
 *  Created by Roy Shilkrot on 12/23/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */

#undef __SFM__DEBUG__

#include "FeatureMatching.h"
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>
#include <set>

using namespace std;
using namespace cv;

void KeyPointsToPoints(const vector<KeyPoint>& kps, vector<Point2f>& ps) {
	ps.clear();
	for (unsigned int i=0; i<kps.size(); i++) ps.push_back(kps[i].pt);
}

void MatchFeatures(const Mat& img_1, const Mat& img_1_orig, 
				   const Mat& img_2, const Mat& img_2_orig,
				   const vector<KeyPoint>& imgpts1,
				   const vector<KeyPoint>& imgpts2,
				   const Mat& descriptors_1, 
				   const Mat& descriptors_2,
				   vector<KeyPoint>& fullpts1,
				   vector<KeyPoint>& fullpts2,
				   int strategy,
				   vector<DMatch>* matches) {
	//strategy
	bool use_features_for_matching =		(strategy & STRATEGY_USE_FEATURE_MATCH) > 0;
	bool use_optical_flow_for_matching =	(strategy & STRATEGY_USE_OPTICAL_FLOW) > 0;
	bool use_dense_optflow =				(strategy & STRATEGY_USE_DENSE_OF) > 0;
	bool use_horiz_disparity =				(strategy & STRATEGY_USE_HORIZ_DISPARITY) > 0;
	
	
	std::vector< DMatch > good_matches_,very_good_matches_;
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	
	Mat_<Point2f> flow_from_features(img_1.size());
#ifdef __SFM__DEBUG__
	Mat outputflow; img_1_orig.copyTo(outputflow);
#endif		
	
	bool update_imgpts1 = (imgpts1.size()<=0);
	bool update_imgpts2 = (imgpts2.size()<=0);
	
	cout << "----------------------------------------------------------------------"<<endl;
	if (update_imgpts1) {
		cout << "imgpts1 empty, get new" << endl;
	} else {
		cout << "imgpts1 has " << imgpts1.size() << " points (descriptors " << descriptors_1.rows << ")" << endl;
	}
	if (update_imgpts2) {
		cout << "imgpts2 empty, get new" << endl;
	} else {
		cout << "imgpts2 has " << imgpts2.size() << " points (descriptors " << descriptors_2.rows << ")" << endl;
	}
	
	if(use_features_for_matching) 
	{	
		//-- Step 1: Detect the keypoints using SURF Detector
		int minHessian = 10;
		
		//		GridAdaptedFeatureDetector detector(new SurfFeatureDetector(minHessian), 1000,1,1);
		SurfFeatureDetector detector( minHessian );
		
		if(update_imgpts1) {
			detector.detect( img_1, keypoints_1 );
		} else {
			keypoints_1 = imgpts1;
		}

		if(update_imgpts2) {
			detector.detect( img_2, keypoints_2 );
		} else {
			keypoints_2 = imgpts2;
		}

		
		//-- Step 2: Calculate descriptors (feature vectors)
		//		SurfDescriptorExtractor extractor(8,4,true);
		SiftDescriptorExtractor extractor(48,16,true);
		//	OpponentColorDescriptorExtractor extractor(new SurfDescriptorExtractor);
		
		if(descriptors_1.empty()) {
//			Mat desc;
//			extractor.compute( img_1, keypoints_1, desc );
//			desc.copyTo(descriptors_1);
			CV_Error(0,"descriptors_1 is empty");
		}
		if(descriptors_2.empty()) {
//			Mat desc;
//			extractor.compute( img_2, keypoints_2, desc );
//			desc.copyTo(descriptors_2);
			CV_Error(0,"descriptors_2 is empty");
		}
		
		//-- Step 3: Matching descriptor vectors using FLANN matcher
		//FlannBasedMatcher matcher;
		BruteForceMatcher<L2<float> > matcher;
		std::vector< DMatch > matches_;
		if (matches == NULL) {
			matches = &matches_;
		}
		if (matches->size() == 0) {
#ifdef __SFM__DEBUG__
			cout << "matching desc1="<<descriptors_1.rows<<", desc2="<<descriptors_2.rows<<endl;
#endif
			matcher.match( descriptors_1, descriptors_2, *matches );
		}
		
#ifdef __SFM__DEBUG__
		cout << "matches->size() " << matches->size() << endl;
#endif
		
		double max_dist = 0; double min_dist = 1000.0;
		//-- Quick calculation of max and min distances between keypoints
		for(unsigned int i = 0; i < matches->size(); i++ )
		{ 
			double dist = (*matches)[i].distance;
			if( dist < min_dist ) min_dist = dist;
			if( dist > max_dist ) max_dist = dist;
		}
		
#ifdef __SFM__DEBUG__
		printf("-- Max dist : %f \n", max_dist );
		printf("-- Min dist : %f \n", min_dist );
#endif
		
		vector<KeyPoint> imgpts1_good,imgpts2_good;

		if (min_dist <= 0) {
			min_dist = 10.0;
		}
		
		double cutoff = 4.0*min_dist;
		std::set<int> existing_trainIdx;
		for(unsigned int i = 0; i < matches->size(); i++ )
		{ 
			if ((*matches)[i].trainIdx <= 0) {
				(*matches)[i].trainIdx = (*matches)[i].imgIdx;
			}
			
			if( existing_trainIdx.find((*matches)[i].trainIdx) == existing_trainIdx.end() && 
				(*matches)[i].trainIdx >= 0 && (*matches)[i].trainIdx < (int)(keypoints_2.size()) &&
				(*matches)[i].distance > 0.0 && (*matches)[i].distance < cutoff ) 
			{
				good_matches_.push_back( (*matches)[i]);
				imgpts1_good.push_back(keypoints_1[(*matches)[i].queryIdx]);
				imgpts2_good.push_back(keypoints_2[(*matches)[i].trainIdx]);
				existing_trainIdx.insert((*matches)[i].trainIdx);
			}
		}
		
		
#ifdef __SFM__DEBUG__
		cout << "keypoints_1.size() " << keypoints_1.size() << " imgpts1_good.size() " << imgpts1_good.size() << endl;
		cout << "keypoints_2.size() " << keypoints_2.size() << " imgpts2_good.size() " << imgpts2_good.size() << endl;

		{
			//-- Draw only "good" matches
			Mat img_matches;
			drawMatches( img_1, keypoints_1, img_2, keypoints_2,
						good_matches_, img_matches, Scalar::all(-1), Scalar::all(-1),
						vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );		
			//-- Show detected matches
			imshow( "Feature Matches", img_matches );
			waitKey(100);
			destroyWindow("Feature Matches");
		}
#endif
		
		//Let the feature matching guide the general flow...
		
		vector<uchar> status;
		vector<KeyPoint> imgpts2_very_good,imgpts1_very_good;
		
		
		//Select features that make epipolar sense
		{
			vector<Point2f> pts1,pts2;
			KeyPointsToPoints(imgpts1_good, pts1);
			KeyPointsToPoints(imgpts2_good, pts2);
#ifdef __SFM__DEBUG__
			cout << "pts1 " << pts1.size() << " (orig pts " << imgpts1_good.size() << ")" << endl;
			cout << "pts2 " << pts2.size() << " (orig pts " << imgpts2_good.size() << ")" << endl;
#endif
			Mat F = findFundamentalMat(pts1, pts2, FM_RANSAC, 0.1, 0.99, status);
		}
		cout << "Fundamental mat is keeping " << countNonZero(status) << " / " << status.size() << endl;	
		
		double status_nz = countNonZero(status); 
		double status_sz = status.size();
		double kept_ratio = status_nz / status_sz;
		if (kept_ratio > 0.2) {
			
			for (unsigned int i=0; i<imgpts1_good.size(); i++) {
				if (status[i]) 
				{
					imgpts1_very_good.push_back(imgpts1_good[i]);
					imgpts2_very_good.push_back(imgpts2_good[i]);
				}
			}
			
			if(use_optical_flow_for_matching) {
				//Estimate the overall 2D homography
		//		Mat_<double> H;
		//		{
		//			vector<Point2f> pts1,pts2;
		//			KeyPointsToPoints(imgpts1_very_good, pts1);
		//			KeyPointsToPoints(imgpts2_very_good, pts2);
		//			cout << "pts1 " << pts1.size() << endl;
		//			cout << "pts2 " << pts2.size() << endl;
		//			H = findHomography(pts1, pts2, CV_RANSAC, 0.001);
		//			cout << "homography from features " << endl << H << endl;
		//		}
				Mat_<double> T;
				{
					vector<Point2f> pts1,pts2;
					KeyPointsToPoints(imgpts1_very_good, pts1);
					KeyPointsToPoints(imgpts2_very_good, pts2);
#ifdef __SFM__DEBUG__
					cout << "pts1 " << pts1.size() << endl;
					cout << "pts2 " << pts2.size() << endl;
#endif
					T = estimateRigidTransform(pts1,pts2, false);
#ifdef __SFM__DEBUG__
					cout << "rigid transform from features " << endl << T << endl;
#endif
				}
				
				//Create the approximate flow using the estimated overall motion
				for (int x=0; x<img_1.cols; x++) {
					for (int y=0; y<img_1.rows; y++) {
		//				Mat_<double> moved = H * (Mat_<double>(3,1) << x , y , 1);
						Mat_<double> moved = T * (Mat_<double>(3,1) << x , y , 1);
						Point2f movedpt(moved(0),moved(1));
						flow_from_features(y,x) = Point2f(movedpt.x-x,movedpt.y-y);
#ifdef __SFM__DEBUG__
		//				circle(outputflow, Point(x,y), 1, Scalar(0,255*norm(flow_from_features(y,x))/250), 1);
						if (x%20 == 0 && y%20 == 0) {
		//					cout << "Point " << Point(x,y) << " moved to " << movedpt << endl;
							line(outputflow, Point(x,y), movedpt, Scalar(0,255*norm(flow_from_features(y,x))/50), 1);
						}
#endif
					}
				}
#ifdef __SFM__DEBUG__
				imshow("flow", outputflow);
				waitKey(100);
				destroyWindow("flow");
#endif		
			}
		} 
	} 
	if(use_optical_flow_for_matching) 
	{
#ifdef __SFM__DEBUG__
		img_1_orig.copyTo(outputflow);
#endif		
		double t = getTickCount();
		cout << "Optical Flow...";
		if(use_dense_optflow) {
			cout << "Dense...";
			Mat_<Point2f> _flow,flow;
			if (use_features_for_matching) {
				flow_from_features.copyTo(flow);
			} else {
				//coarse
				calcOpticalFlowFarneback(img_1,img_2,flow,0.5,5,150,60,7,1.5,OPTFLOW_FARNEBACK_GAUSSIAN);
			}

			//refine
			calcOpticalFlowFarneback(img_1,img_2,flow,0.5,2,40,40,5,0.5,OPTFLOW_USE_INITIAL_FLOW);
			calcOpticalFlowFarneback(img_1,img_2,flow,0.5,0,25,40,3,0.25,OPTFLOW_USE_INITIAL_FLOW);
			
			//imgpts1.clear(); imgpts2.clear(); 
			good_matches_.clear(); keypoints_1.clear(); keypoints_2.clear();
			
			for (int x=0;x<flow.cols; x+=1) {
				for (int y=0; y<flow.rows; y+=1) {
					if (norm(flow(y,x)) < 20 || norm(flow(y,x)) > 100) {
						continue; //discard points that havn't moved
					}
					Point2f p(x,y),p1(x+flow(y,x).x,y+flow(y,x).y);
					
					//line(outputflow, p, p1, Scalar(0,255*norm(flow(y,x))/50), 1);
#ifdef __SFM__DEBUG__
					circle(outputflow, p, 1, Scalar(0,255*norm(flow(y,x))/50), 1);
#endif
					if (x%10 == 0 && y%10 == 0) {
//						imgpts1.push_back(KeyPoint(p,1));
//						imgpts2.push_back(KeyPoint(p1,1));
						good_matches_.push_back(DMatch(imgpts1.size()-1,imgpts1.size()-1,1.0));
						keypoints_1.push_back(KeyPoint(p,1));
						keypoints_2.push_back(KeyPoint(p1,1));
					}
					fullpts1.push_back(KeyPoint(p,1));
					fullpts2.push_back(KeyPoint(p1,1));
				}
			}		
		} else {
			vector<Point2f> corners,nextPts; vector<uchar> status; vector<float> err;
			goodFeaturesToTrack(img_1, corners, 2000, 0.001, 10);
			cornerSubPix(img_1, corners, Size(15,15), Size(-1,-1), TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001 ));
			calcOpticalFlowPyrLK(img_1, img_2, corners, nextPts, status, err, Size(45,45));
			for (unsigned int i=0; i<corners.size(); i++) {
				if(status[i] == 1) {
#ifdef __SFM__DEBUG__
					line(outputflow, corners[i], nextPts[i], Scalar(0,255), 1);
#endif
//					imgpts1.push_back(KeyPoint(corners[i],1));
//					imgpts2.push_back(KeyPoint(nextPts[i],1));
					good_matches_.push_back(DMatch(imgpts1.size()-1,imgpts1.size()-1,1.0));
					keypoints_1.push_back(KeyPoint(corners[i],1));
					keypoints_2.push_back(KeyPoint(nextPts[i],1));
				}
			}
		}
		t = ((double)getTickCount() - t)/getTickFrequency();
		cout << "Done. (" << t <<"s)"<< endl;
#ifdef __SFM__DEBUG__
		imshow("flow", outputflow);
		waitKey(100);
		destroyWindow("flow");
#endif
	} 
	else if(use_horiz_disparity) 
	{		
		double downscale = 0.6;
		Mat small_im1; resize(img_1_orig,small_im1,Size(),downscale,downscale);
		Mat small_im2; resize(img_2_orig,small_im2,Size(),downscale,downscale);
		int numberOfDisparities = ((small_im1.cols/8) + 15) & -16;
		
		StereoSGBM sgbm;
		sgbm.preFilterCap = 63;
		sgbm.SADWindowSize = 3;
		
		int cn = img_1_orig.channels();
		
		sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
		sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
		sgbm.minDisparity = 0;
		sgbm.numberOfDisparities = numberOfDisparities;
		sgbm.uniquenessRatio = 10;
		sgbm.speckleWindowSize = 100;
		sgbm.speckleRange = 32;
		sgbm.disp12MaxDiff = 1;
		sgbm.fullDP = false;
		
		Mat_<short> disp;
		sgbm(small_im1, small_im2, disp);
		Mat disp8; disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));
#ifdef __SFM__DEBUG__
		imshow("disparity",disp8);
		waitKey(0);
		destroyWindow("disparity");
#endif		
		Mat outputflow; img_1_orig.copyTo(outputflow);
		Mat_<short> disp_orig_scale; resize(disp,disp_orig_scale,img_1.size());
		
		for (int x=0;x<disp_orig_scale.cols; x+=1) {
			for (int y=0; y<disp_orig_scale.rows; y+=1) {
				float _d = ((float)disp_orig_scale(y,x))/(16.0 * downscale);
				if (fabsf(_d) > 150.0f || fabsf(_d) < 5.0f) {
					continue; //discard strange points 
				}
				Point2f p(x,y),p1(x-_d,y);
#ifdef __SFM__DEBUG__
				circle(outputflow, p, 1, Scalar(0,255*_d/50.0), 1);
#endif
				if (x%10 == 0 && y%10 == 0) {
//					imgpts1.push_back(KeyPoint(p,1));
//					imgpts2.push_back(KeyPoint(p1,1));
					good_matches_.push_back(DMatch(imgpts1.size()-1,imgpts1.size()-1,1.0));
					keypoints_1.push_back(KeyPoint(p,1));
					keypoints_2.push_back(KeyPoint(p1,1));
				}
				fullpts1.push_back(KeyPoint(p,1));
				fullpts2.push_back(KeyPoint(p1,1));
			}
		}		
#ifdef __SFM__DEBUG__		
		imshow("outputflow", outputflow);
		waitKey(0);
		destroyWindow("outputflow");
#endif
	}
	
	//Draw matches
//	if(0) 
#ifdef __SFM__DEBUG__
	{
		//-- Draw only "good" matches
		Mat img_matches;
		drawMatches( img_1, keypoints_1, img_2, keypoints_2,
					good_matches_, img_matches, Scalar::all(-1), Scalar::all(-1),
					vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );		
		//-- Show detected matches
		imshow( "Good Matches", img_matches );
		waitKey(100);
		destroyWindow("Good Matches");
	}
#endif
}