/*
 *  Visualization.h
 *  EyeRingOpenCV
 *
 *  Created by Roy Shilkrot on 12/23/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

void RunVisualization(const std::vector<cv::Point3d>& pointcloud,
					  const std::vector<cv::Vec3b>& pointcloud_RGB = std::vector<cv::Vec3b>(),
					  const cv::Mat& img_1_orig = cv::Mat(), 
					  const cv::Mat& img_2_orig = cv::Mat(),
					  const std::vector<cv::KeyPoint>& correspImg1Pt = std::vector<cv::KeyPoint>());
