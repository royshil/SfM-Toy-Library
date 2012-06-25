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
					  const std::vector<cv::Point3d>& pointcloud1 = std::vector<cv::Point3d>(),
					  const std::vector<cv::Vec3b>& pointcloud1_RGB = std::vector<cv::Vec3b>());
