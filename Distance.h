/*
 *  Distance.h
 *  EyeRingOpenCV
 *
 *  Created by Roy Shilkrot on 1/1/12.
 *  Copyright 2012 MIT. All rights reserved.
 *
 */

#include <opencv2/opencv.hpp>
#include <vector>

#include "Triangulation.h"
#include "FeatureMatching.h"
#include "FindCameraMatrices.h"

class Distance {
private:
	std::vector<cv::Point2d> imgpts1,
							imgpts2,
							fullpts1,
							fullpts2,
							imgpts1_good,
							imgpts2_good;
	cv::Mat left_im,
			left_im_orig,
			right_im,
			right_im_orig;
	cv::Matx34d P,P1;
	cv::Mat K;
	cv::Mat_<double> Kinv;

	cv::Mat cam_matrix,distortion_coeff;
	
	std::vector<cv::Point3d> pointcloud;
	std::vector<cv::Point> correspImg1Pt;
	
	bool features_matched;
public:
	const std::vector<cv::Point3d>& getpointcloud() { return pointcloud; }
	const cv::Mat& getleft_im_orig() { return left_im_orig; }
	const cv::Mat& getright_im_orig() { return right_im_orig; }
	const std::vector<cv::Point>& getcorrespImg1Pt() { return correspImg1Pt; }
	
		//c'tor
	Distance(const cv::Mat& left_im_, const cv::Mat& right_im_):
		features_matched(false)
	{
		left_im_.copyTo(left_im);
		right_im_.copyTo(right_im);
		left_im.copyTo(left_im_orig);
		cvtColor(left_im_orig, left_im, CV_BGR2GRAY);
		right_im.copyTo(right_im_orig);
		cvtColor(right_im_orig, right_im, CV_BGR2GRAY);
		
		P = cv::Matx34d(1,0,0,0,
						0,1,0,0,
						0,0,1,0);
		P1 = cv::Matx34d(1,0,0,50,
						 0,1,0,0,
						 0,0,1,0);

		cv::FileStorage fs;
		fs.open("../../Calibration/out_camera_data.yml",cv::FileStorage::READ);
		fs["camera_matrix"]>>cam_matrix;
		fs["distortion_coefficients"]>>distortion_coeff;

		K = cam_matrix;
		invert(K, Kinv); //get inverse of camera matrix
	}
	
	void OnlyMatchFeatures(int strategy = STRATEGY_USE_OPTICAL_FLOW + STRATEGY_USE_DENSE_OF + STRATEGY_USE_FEATURE_MATCH) {
		imgpts1.clear(); imgpts2.clear(); fullpts1.clear(); fullpts2.clear();
		MatchFeatures(left_im, left_im_orig, 
					  right_im, right_im_orig,
					  imgpts1,
					  imgpts2,
					  fullpts1,
					  fullpts2,
					  strategy);
		
		features_matched = true;
	}
	
	void RecoverDepthFromImages() {			
		
		if(!features_matched) 
			OnlyMatchFeatures();
		
		FindCameraMatrices(K, Kinv, imgpts1, imgpts2, imgpts1_good, imgpts2_good, P, P1
#ifdef __SFM__DEBUG__
						   ,left_im,right_im
#endif
						   );
		
		//TODO: if the P1 matrix is far away from identity rotation - the solution is probably invalid...
		//so use an identity matrix
		
		std::vector<cv::Point2d>& pt_set1 = (fullpts1.size()>0) ? fullpts1 : imgpts1_good;
		std::vector<cv::Point2d>& pt_set2 = (fullpts2.size()>0) ? fullpts2 : imgpts2_good;
		
#if 0
		//Use OpenCVs cvCorrectMatches
		{
			Mat m_F(F); CvMat cvm_F = m_F;
			Mat m_Pts1(pt_set1); CvMat cvm_Pts1 = Mat(m_Pts1.t());
			Mat m_Pts2(pt_set2); CvMat cvm_Pts2 = Mat(m_Pts2.t());
			cvCorrectMatches(&cvm_F, &cvm_Pts1, &cvm_Pts2, NULL, NULL);
			Mat m_Pts1_result(&cvm_Pts1); Mat(m_Pts1_result.t()).copyTo(m_Pts1);
			Mat m_Pts2_result(&cvm_Pts2); Mat(m_Pts2_result.t()).copyTo(m_Pts2);
		}
#endif
		
		{
		//	undistortPoints(pt_set1, pt_set1, cam_matrix, distortion_coeff);
		//	undistortPoints(pt_set2, pt_set2, cam_matrix, distortion_coeff);
		//	undistortPoints(pt_set1, pt_set1, cam_matrix, Mat_<double>::zeros(1,4));
		//	undistortPoints(pt_set2, pt_set2, cam_matrix, Mat_<double>::zeros(1,4));
		//	undistortPoints(pt_set1, pt_set1, Mat_<double>::eye(3,3), distortion_coeff);
		//	undistortPoints(pt_set2, pt_set2, Mat_<double>::eye(3,3), distortion_coeff);
		}
		
		TriangulatePoints(pt_set1, pt_set2, Kinv, P, P1, pointcloud, correspImg1Pt);
	}
};