/*
 *  Common.h
 *  SfMToyExample
 *
 *  Created by Roy Shilkrot on 4/27/12.
 *  Copyright 2012 MIT. All rights reserved.
 *
 */
#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

struct CloudPoint {
	cv::Point3d pt;
	std::vector<int> idx_in_imgpts_for_img;
};

void KeyPointsToPoints(const std::vector<cv::KeyPoint>& kps, std::vector<cv::Point2f>& ps);

std::vector<cv::Point3d> CloudPointsToPoints(const std::vector<CloudPoint> cpts);

void GetAlignedPointsFromMatch(const std::vector<cv::KeyPoint>& imgpts1,
							   const std::vector<cv::KeyPoint>& imgpts2,
							   const std::vector<cv::DMatch>& matches,
							   std::vector<cv::KeyPoint>& pt_set1,
							   std::vector<cv::KeyPoint>& pt_set2);