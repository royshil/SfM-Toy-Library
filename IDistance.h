/*
 *  IDistance.h
 *  SfMToyExample
 *
 *  Created by Roy Shilkrot on 4/15/12.
 *  Copyright 2012 MIT. All rights reserved.
 *
 */

#pragma once

#define STRATEGY_USE_OPTICAL_FLOW		1
#define STRATEGY_USE_DENSE_OF			2
#define STRATEGY_USE_FEATURE_MATCH		4
#define STRATEGY_USE_HORIZ_DISPARITY	8

class IDistance {
public:
	virtual void OnlyMatchFeatures(int strategy = STRATEGY_USE_OPTICAL_FLOW + STRATEGY_USE_DENSE_OF + STRATEGY_USE_FEATURE_MATCH) = 0;
	virtual void RecoverDepthFromImages() = 0;
	virtual std::vector<cv::Point3d> getPointCloud() = 0;
	virtual const std::vector<cv::Vec3b>& getPointCloudRGB() = 0;
};