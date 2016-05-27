/*
 *  Distance.h
 *  SfMToyLibrary
 *
 *  Created by Roy Shilkrot on 1/1/12.
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2013 Roy Shilkrot
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
 *
 */
#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "Common.h"
#include "IDistance.h"
#include "Triangulation.h"
#include "FindCameraMatrices.h"
#include "RichFeatureMatcher.h"

class Distance : public IDistance {
private:
	std::vector<cv::KeyPoint> imgpts1,
							imgpts2,
							fullpts1,
							fullpts2,
							imgpts1_good,
							imgpts2_good;
	cv::Mat descriptors_1; 
	cv::Mat descriptors_2;
	
	cv::Mat left_im,
			left_im_orig,
			right_im,
			right_im_orig;
	cv::Matx34d P,P1;
	cv::Mat K;
	cv::Mat_<double> Kinv;

	cv::Mat cam_matrix,distortion_coeff;
	
	std::vector<CloudPoint> pointcloud;
	std::vector<cv::KeyPoint> correspImg1Pt;
	
	bool features_matched;
public:
	std::vector<cv::Point3d> getPointCloud() { return CloudPointsToPoints(pointcloud); }
	const cv::Mat& getleft_im_orig() { return left_im_orig; }
	const cv::Mat& getright_im_orig() { return right_im_orig; }
	const std::vector<cv::KeyPoint>& getcorrespImg1Pt() { return correspImg1Pt; }
	const std::vector<cv::Vec3b>& getPointCloudRGB() { return std::vector<cv::Vec3b>();}
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
		fs.open("../out_camera_data.yml",cv::FileStorage::READ);
		fs["camera_matrix"]>>cam_matrix;
		fs["distortion_coefficients"]>>distortion_coeff;

		K = cam_matrix;
		invert(K, Kinv); //get inverse of camera matrix
	}
	
	void OnlyMatchFeatures(int strategy = STRATEGY_USE_OPTICAL_FLOW + STRATEGY_USE_DENSE_OF + STRATEGY_USE_FEATURE_MATCH) {
		imgpts1.clear(); imgpts2.clear(); fullpts1.clear(); fullpts2.clear();
		
		std::vector<cv::Mat> imgs; imgs.push_back(left_im); imgs.push_back(right_im);
		std::vector<std::vector<cv::KeyPoint> > imgpts; imgpts.push_back(imgpts1); imgpts.push_back(imgpts2);
		
		RichFeatureMatcher rfm(imgs,imgpts);
		rfm.MatchFeatures(0, 1);
		
		imgpts1 = rfm.GetImagePoints(0);
		imgpts2 = rfm.GetImagePoints(1);
		
		features_matched = true;
	}
	
	void RecoverDepthFromImages() {			
		
		if(!features_matched) 
			OnlyMatchFeatures();
		
		std::vector<cv::DMatch> matches;
		FindCameraMatrices(K, Kinv, distortion_coeff, imgpts1, imgpts2, imgpts1_good, imgpts2_good, P, P1, matches, pointcloud
#ifdef __SFM__DEBUG__
						   ,left_im,right_im
#endif
						   );
		
		//TODO: if the P1 matrix is far away from identity rotation - the solution is probably invalid...
		//so use an identity matrix
		
		std::vector<cv::KeyPoint> pt_set1,pt_set2;
		GetAlignedPointsFromMatch(imgpts1,imgpts2,matches,pt_set1,pt_set2);
		
		TriangulatePoints(pt_set1, pt_set2, K, Kinv,distortion_coeff, P, P1, pointcloud, correspImg1Pt);
	}
};
