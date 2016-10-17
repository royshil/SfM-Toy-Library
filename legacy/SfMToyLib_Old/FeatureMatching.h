/*
 *  FeatureMatching.h
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

#include "IDistance.h"

#include "Common.h"

void MatchFeatures(const cv::Mat& img_1, const cv::Mat& img_1_orig, 
				   const cv::Mat& img_2, const cv::Mat& img_2_orig,
				   const std::vector<cv::KeyPoint>& imgpts1,
				   const std::vector<cv::KeyPoint>& imgpts2,
				   const cv::Mat& descriptors_1, 
				   const cv::Mat& descriptors_2,
				   std::vector<cv::KeyPoint>& fullpts1,
				   std::vector<cv::KeyPoint>& fullpts2,
				   int stretegy = STRATEGY_USE_OPTICAL_FLOW + STRATEGY_USE_DENSE_OF + STRATEGY_USE_FEATURE_MATCH,
				   std::vector<cv::DMatch>* matches = NULL);
