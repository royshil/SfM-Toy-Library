/*
 *  BundleAdjuster.h
 *  ExploringSfMWithOpenCV
 *
 *  Created by Roy Shilkrot on 5/18/12.
 *  Copyright 2012 MIT. All rights reserved.
 *
 */

#include <vector>
#include <opencv2/core/core.hpp>
#include "Common.h"

class BundleAdjuster {
public:
	void adjustBundle(std::vector<CloudPoint>& pointcloud, 
					  const cv::Mat& cam_matrix,
					  const std::vector<std::vector<cv::KeyPoint> >& imgpts,
					  std::map<std::pair<int,int> ,cv::Matx34d>& Pmats);
};