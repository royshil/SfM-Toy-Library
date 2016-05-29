/*
*  MultiCameraPnP.h
*  SfMToyExample
*
*  Created by Roy Shilkrot on 4/27/12.
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2013 Roy Shilkrot
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
*
*/
#pragma once

#include "MultiCameraDistance.h"
#include "Common.h"
#include "SfMUpdateListener.h"

class MultiCameraPnP : public MultiCameraDistance {
	std::vector<CloudPoint> pointcloud_beforeBA;
	std::vector<cv::Vec3b> pointCloudRGB_beforeBA;

public:
	MultiCameraPnP() {}
	MultiCameraPnP(
		const std::vector<cv::Mat>& imgs_, 
		const std::vector<std::string>& imgs_names_, 
		const std::string& imgs_path_):
	MultiCameraDistance(imgs_,imgs_names_,imgs_path_) 
	{
	}

	virtual void RecoverDepthFromImages();

	std::vector<cv::Point3d> getPointCloudBeforeBA() { return CloudPointsToPoints(pointcloud_beforeBA); }
	const std::vector<cv::Vec3b>& getPointCloudRGBBeforeBA() { return pointCloudRGB_beforeBA; }

private:
	void PruneMatchesBasedOnF();
	void AdjustCurrentBundle();
	void GetBaseLineTriangulation();
	void Find2D3DCorrespondences(int working_view, 
		std::vector<cv::Point3f>& ppcloud, 
		std::vector<cv::Point2f>& imgPoints);
	bool FindPoseEstimation(
		int working_view,
		cv::Mat_<double>& rvec,
		cv::Mat_<double>& t,
		cv::Mat_<double>& R,
		std::vector<cv::Point3f> ppcloud,
		std::vector<cv::Point2f> imgPoints);
	bool TriangulatePointsBetweenViews(
		int working_view, 
		int second_view,
		std::vector<struct CloudPoint>& new_triangulated,
		std::vector<int>& add_to_cloud
		);
	
	int FindHomographyInliers2Views(int vi, int vj);
	int m_first_view;
	int m_second_view; //baseline's second view other to 0
	std::set<int> done_views;
	std::set<int> good_views;
	
/********** Subject / Objserver **********/
	std::vector < SfMUpdateListener * > listeners;
public:
    void attach(SfMUpdateListener *sul)
    {
        listeners.push_back(sul);
    }
private:
    void update()
    {
        for (int i = 0; i < listeners.size(); i++)
			listeners[i]->update(getPointCloud(),
								 getPointCloudRGB(),
								 getPointCloudBeforeBA(),
								 getPointCloudRGBBeforeBA(),
								 getCameras());
    }
};
