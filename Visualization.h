/*
 *  Visualization.h
 *  EyeRingOpenCV
 *
 *  Created by Roy Shilkrot on 12/23/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */

#include <opencv2/core/core.hpp>
#include <vector>

void RunVisualization(const std::vector<cv::Point3d>& pointcloud,
					  const cv::Mat& img_1_orig, 
					  const cv::Mat& img_2_orig,
					  const std::vector<cv::Point>& correspImg1Pt);
