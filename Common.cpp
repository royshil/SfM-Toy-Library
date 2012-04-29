/*
 *  Common.cpp
 *  SfMToyExample
 *
 *  Created by Roy Shilkrot on 4/27/12.
 *  Copyright 2012 MIT. All rights reserved.
 *
 */

#include "Common.h"

#include <iostream>

using namespace std;
using namespace cv;

std::vector<cv::Point3d> CloudPointsToPoints(const std::vector<CloudPoint> cpts) {
	std::vector<cv::Point3d> out;
	for (unsigned int i=0; i<cpts.size(); i++) {
		out.push_back(cpts[i].pt);
	}
	return out;
}

void GetAlignedPointsFromMatch(const std::vector<cv::KeyPoint>& imgpts1,
							   const std::vector<cv::KeyPoint>& imgpts2,
							   const std::vector<cv::DMatch>& matches,
							   std::vector<cv::KeyPoint>& pt_set1,
							   std::vector<cv::KeyPoint>& pt_set2) {
//	cout << "imgpts1.size() " << imgpts1.size() << " imgpts2.size() " << imgpts2.size() <<endl;
	for (unsigned int i=0; i<matches.size(); i++) {
//		cout << "matches[i].queryIdx " << matches[i].queryIdx << " matches[i].trainIdx " << matches[i].trainIdx << endl;
		pt_set1.push_back(imgpts1[matches[i].queryIdx]);
		pt_set2.push_back(imgpts2[matches[i].trainIdx]);
	}	
}