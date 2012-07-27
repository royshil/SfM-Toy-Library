/*
 *  MultiCameraDistance.cpp
 *  SfMToyExample
 *
 *  Created by Roy Shilkrot on 3/27/12.
 *  Copyright 2012 MIT. All rights reserved.
 *
 */

#include "MultiCameraDistance.h"
#include "RichFeatureMatcher.h"
#include "OFFeatureMatcher.h"
#include "GPUSURFFeatureMatcher.h"

//c'tor
MultiCameraDistance::MultiCameraDistance(const std::vector<cv::Mat>& imgs_, const std::vector<std::string>& imgs_names_, const std::string& imgs_path_):
imgs_names(imgs_names_),features_matched(false),use_rich_features(true),use_gpu(true)
{		
	std::cout << "=========================== Load Images ===========================\n";
	//ensure images are CV_8UC3
	for (unsigned int i=0; i<imgs_.size(); i++) {
		imgs_orig.push_back(cv::Mat());
		if (!imgs_[i].empty()) {
			if (imgs_[i].type() == CV_8UC1) {
				cvtColor(imgs_[i], imgs_orig[i], CV_GRAY2BGR);
			} else if (imgs_[i].type() == CV_32FC3 || imgs_[i].type() == CV_64FC3) {
				imgs_[i].convertTo(imgs_orig[i],CV_8U,255.0);
			} else {
				imgs_[i].copyTo(imgs_orig[i]);
			}
			
		}
		
		imgs.push_back(cv::Mat());
		cvtColor(imgs_orig[i],imgs[i], CV_BGR2GRAY);
		
		imgpts.push_back(std::vector<cv::KeyPoint>());
		imgpts_good.push_back(std::vector<cv::KeyPoint>());
		std::cout << ".";
	}
	std::cout << std::endl;
		
	//load calibration matrix
	cv::FileStorage fs;
	if(fs.open(imgs_path_+ "\\out_camera_data.yml",cv::FileStorage::READ)) {
		fs["camera_matrix"]>>cam_matrix;
		fs["distortion_coefficients"]>>distortion_coeff;
	} else {
		//no calibration matrix file - mockup calibration
		cv::Size imgs_size = imgs_[0].size();
		cam_matrix = (cv::Mat_<double>(3,3) << imgs_size.height , 0 , imgs_size.width/2.0,
					  0, imgs_size.height, imgs_size.height/2.0,
					  0, 0, 1);
		distortion_coeff = cv::Mat_<double>::zeros(1,4);
	}
	
	K = cam_matrix;
	invert(K, Kinv); //get inverse of camera matrix
}

void MultiCameraDistance::OnlyMatchFeatures(int strategy) 
{
	if(features_matched) return;
	
	if (use_rich_features) {
		if (use_gpu) {
			feature_matcher = new GPUSURFFeatureMatcher(imgs,imgpts);
		} else {
			feature_matcher = new RichFeatureMatcher(imgs,imgpts);
		}
	} else {
		feature_matcher = new OFFeatureMatcher(imgs,imgpts);
	}	

	if(strategy & STRATEGY_USE_OPTICAL_FLOW)
		use_rich_features = false;

	int loop1_top = imgs.size() - 1, loop2_top = imgs.size();
	int frame_num_i = 0;
	//#pragma omp parallel for schedule(dynamic)
	
	if (use_rich_features) {
		for (frame_num_i = 0; frame_num_i < loop1_top; frame_num_i++) {
			for (int frame_num_j = frame_num_i + 1; frame_num_j < loop2_top; frame_num_j++)
			{
				std::vector<cv::KeyPoint> fp,fp1;
				std::cout << "------------ Match " << imgs_names[frame_num_i] << ","<<imgs_names[frame_num_j]<<" ------------\n";
				std::vector<cv::DMatch> matches_tmp;
				feature_matcher->MatchFeatures(frame_num_i,frame_num_j,&matches_tmp);
				
				//#pragma omp critical
				{
					matches_matrix[std::make_pair(frame_num_i,frame_num_j)] = matches_tmp;
				}
			}
		}
	} else {
		for (frame_num_i = 0; frame_num_i < loop1_top; frame_num_i++) {
			for (int frame_num_j = frame_num_i + 1; frame_num_j < loop2_top; frame_num_j++)
			{
				std::cout << "------------ Match " << imgs_names[frame_num_i] << ","<<imgs_names[frame_num_j]<<" ------------\n";
				std::vector<cv::DMatch> matches_tmp;
				feature_matcher->MatchFeatures(frame_num_i,frame_num_j,&matches_tmp);
				matches_matrix[std::make_pair(frame_num_i,frame_num_j)] = matches_tmp;
			}
		}
	}

	features_matched = true;
}

bool MultiCameraDistance::CheckCoherentRotation(cv::Mat_<double>& R) {
//	std::cout << "R; " << R << std::endl;
	double s = cv::norm(R,cv::Mat_<double>::eye(3,3),cv::NORM_L1);
//	std::cout << "Distance from I: " << s << std::endl;
	if (s > 2.3) { // norm of R from I is large -> probably bad rotation
		std::cout << "rotation is probably not coherent.." << std::endl;
		return false;	//skip triangulation
	}
	return true;
}
