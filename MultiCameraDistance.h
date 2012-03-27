/*
 *  MultiCameraDistance.h
 *  SfMToyExample
 *
 *  Created by Roy Shilkrot on 3/27/12.
 *  Copyright 2012 MIT. All rights reserved.
 *
 */

#include <opencv2/opencv.hpp>
#include <vector>

#include "Triangulation.h"
#include "FeatureMatching.h"
#include "FindCameraMatrices.h"

class MultiCameraDistance {	
private:
	std::vector<std::vector<cv::Point2d> > imgpts;
	std::vector<std::vector<cv::Point2d> > fullpts;
	std::vector<std::vector<cv::Point2d> > imgpts1_good;

	std::vector<cv::Mat> imgs, imgs_orig;

	std::vector<cv::Matx34d> Pmats;

	cv::Mat K;
	cv::Mat_<double> Kinv;
	
	cv::Mat cam_matrix,distortion_coeff;
	
	std::vector<cv::Point3d> pointcloud;
	std::vector<cv::Point> correspImg1Pt;
	
	bool features_matched;
public:
	const std::vector<cv::Point3d>& getpointcloud() { return pointcloud; }
	const cv::Mat& get_im_orig(int frame_num) { return imgs_orig[frame_num]; }
	const std::vector<cv::Point>& getcorrespImg1Pt() { return correspImg1Pt; }
	
	//c'tor
	MultiCameraDistance():features_matched(false)
	{		
		cv::FileStorage fs;
		fs.open("../../Calibration/out_camera_data.yml",cv::FileStorage::READ);
		fs["camera_matrix"]>>cam_matrix;
		fs["distortion_coefficients"]>>distortion_coeff;
		
		K = cam_matrix;
		invert(K, Kinv); //get inverse of camera matrix
	}
	
	void OnlyMatchFeatures(int strategy = STRATEGY_USE_OPTICAL_FLOW + STRATEGY_USE_DENSE_OF + STRATEGY_USE_FEATURE_MATCH) 
	{
		//TODO pair-wise feature matching
		for (int frame_num; frame_num < imgs.size(); frame_num++) {
//		MatchFeatures(left_im, left_im_orig, 
//					  right_im, right_im_orig,
//					  imgpts1,
//					  imgpts2,
//					  fullpts1,
//					  fullpts2,
//					  strategy);
		}
		
		features_matched = true;
	}
	
	void RecoverDepthFromImages() {			
		
		if(!features_matched) 
			OnlyMatchFeatures();
		
		for (int frame_num; frame_num < imgs.size(); frame_num++) {
		
		//TODO: obtain camera matrices from pairwise matches
//		FindCameraMatrices(K, Kinv, imgpts1, imgpts2, imgpts1_good, imgpts2_good, P, P1
//#ifdef __SFM__DEBUG__
//						   ,left_im,right_im
//#endif
//						   );
		
		//TODO: if the P1 matrix is far away from identity rotation - the solution is probably invalid...
		//so use an identity matrix
		
//		std::vector<cv::Point2d>& pt_set1 = (fullpts1.size()>0) ? fullpts1 : imgpts1_good;
//		std::vector<cv::Point2d>& pt_set2 = (fullpts2.size()>0) ? fullpts2 : imgpts2_good;
		
			
		//TODO: triangulate points for each pair, and transform to base frame

//		TriangulatePoints(pt_set1, pt_set2, Kinv, P, P1, pointcloud, correspImg1Pt);
			
		}
	}
};