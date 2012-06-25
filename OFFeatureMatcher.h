/*
 *  OFFeatureMatcher.h
 *  ExploringSfMWithOpenCV
 *
 *  Created by Roy Shilkrot on 5/17/12.
 *  Copyright 2012 MIT. All rights reserved.
 *
 */

#include "IFeatureMatcher.h"

class OFFeatureMatcher : public IFeatureMatcher {
	std::vector<cv::Mat>& imgs; 
	std::vector<std::vector<cv::KeyPoint> >& imgpts;
	
public:
	OFFeatureMatcher(std::vector<cv::Mat>& imgs_, 
					 std::vector<std::vector<cv::KeyPoint> >& imgpts_);
	void MatchFeatures(int idx_i, int idx_j, std::vector<cv::DMatch>* matches);
	std::vector<cv::KeyPoint> GetImagePoints(int idx) { return imgpts[idx]; }
};