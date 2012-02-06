/*
 *  FeatureMatching.h
 *  EyeRingOpenCV
 *
 *  Created by Roy Shilkrot on 12/23/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */

#include <opencv2/core/core.hpp>
#include <vector>

#define STRATEGY_USE_OPTICAL_FLOW		1
#define STRATEGY_USE_DENSE_OF			2
#define STRATEGY_USE_FEATURE_MATCH		4
#define STRATEGY_USE_HORIZ_DISPARITY	8

void MatchFeatures(const cv::Mat& img_1, const cv::Mat& img_1_orig, 
				   const cv::Mat& img_2, const cv::Mat& img_2_orig,
				   std::vector<cv::Point2d>& imgpts1,
				   std::vector<cv::Point2d>& imgpts2,
				   std::vector<cv::Point2d>& fullpts1,
				   std::vector<cv::Point2d>& fullpts2,
				   int stretegy = STRATEGY_USE_OPTICAL_FLOW + STRATEGY_USE_DENSE_OF + STRATEGY_USE_FEATURE_MATCH);
