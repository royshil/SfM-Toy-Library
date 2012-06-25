/*
 *  MultiCameraDistance.h
 *  SfMToyExample
 *
 *  Created by Roy Shilkrot on 3/27/12.
 *  Copyright 2012 MIT. All rights reserved.
 *
 */
#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>

#include "IDistance.h"
#include "Triangulation.h"
#include "IFeatureMatcher.h"
#include "FindCameraMatrices.h"


class MultiCameraDistance  : public IDistance {	
protected:
	std::vector<std::vector<cv::KeyPoint> > imgpts;
	std::vector<std::vector<cv::KeyPoint> > fullpts;
	std::vector<std::vector<cv::KeyPoint> > imgpts_good;

	std::map<std::pair<int,int> ,std::vector<cv::DMatch> > matches_matrix;
	
	std::vector<cv::Mat> imgs, imgs_orig;
	std::vector<std::string> imgs_names;
	
	std::map<std::pair<int,int> ,cv::Matx34d> Pmats;

	cv::Mat K;
	cv::Mat_<double> Kinv;
	
	cv::Mat cam_matrix,distortion_coeff;
	
	std::vector<CloudPoint> pointcloud;
	std::vector<cv::Vec3b> pointCloudRGB;
	std::vector<cv::KeyPoint> correspImg1Pt; //TODO: remove
	
	cv::Ptr<IFeatureMatcher> feature_matcher;
	
	bool features_matched;
	bool use_rich_features;
	bool use_gpu;
public:
	std::vector<cv::Point3d> getPointCloud() { return CloudPointsToPoints(pointcloud); }
	const cv::Mat& get_im_orig(int frame_num) { return imgs_orig[frame_num]; }
	const std::vector<cv::KeyPoint>& getcorrespImg1Pt() { return correspImg1Pt; }
	const std::vector<cv::Vec3b>& getPointCloudRGB() { return pointCloudRGB; }

	MultiCameraDistance(const std::vector<cv::Mat>& imgs_, const std::vector<std::string>& imgs_names_, const std::string& imgs_path_);	
	virtual void OnlyMatchFeatures(int strategy = STRATEGY_USE_FEATURE_MATCH);	
	bool CheckCoherentRotation(cv::Mat_<double>& R);
};
