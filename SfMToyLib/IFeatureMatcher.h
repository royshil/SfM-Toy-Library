/*
 *  IFeatureMatcher.h
 *  ExploringSfMWithOpenCV
 *
 *  Created by Roy Shilkrot on 5/17/12.
 *  Copyright 2012 MIT. All rights reserved.
 *
 */
#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

#include "IDistance.h"

#include "Common.h"


/**
 Feature Matching Interface
 */
class IFeatureMatcher {
public:
	virtual void MatchFeatures(int idx_i, int idx_j, std::vector<cv::DMatch>* matches) = 0;
	virtual std::vector<cv::KeyPoint> GetImagePoints(int idx) = 0;
};