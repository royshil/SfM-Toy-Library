/*
 *  MultiCameraPnP.h
 *  SfMToyExample
 *
 *  Created by Roy Shilkrot on 4/27/12.
 *  Copyright 2012 MIT. All rights reserved.
 *
 */
#pragma once

#include "MultiCameraDistance.h"

#include "Visualization.h"

#include "Common.h"

class MultiCameraPnP : public MultiCameraDistance {
	std::vector<CloudPoint> pointcloud_beforeBA;
	std::vector<cv::Vec3b> pointCloudRGB_beforeBA;

public:
	MultiCameraPnP(const std::vector<cv::Mat>& imgs_, const std::vector<std::string>& imgs_names_, const std::string& imgs_path_):
	MultiCameraDistance(imgs_,imgs_names_,imgs_path_) 
	{
	}
	
	virtual void RecoverDepthFromImages();
	
	std::vector<cv::Point3d> getPointCloudBeforeBA() { return CloudPointsToPoints(pointcloud_beforeBA); }
	const std::vector<cv::Vec3b>& getPointCloudRGBBeforeBA() { return pointCloudRGB_beforeBA; }
	
private:
	void GetBaseLineTriangulation();
	int second_view; //baseline's second view other to 0
	std::vector<CloudPoint> pcloud;
};
