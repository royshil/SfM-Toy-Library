/*
 *  Visualization.cpp
 *  EyeRingOpenCV
 *
 *  Created by Roy Shilkrot on 12/23/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */

#include "Visualization.h"

#include <pcl/common/common.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/io.h>
#include <pcl/io/file_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>

#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;
 
void PopulatePCLPointCloud(const vector<Point3d>& pointcloud, 
						   const std::vector<cv::Vec3b>& pointcloud_RGB, 
						   const Mat& img_1_orig, 
						   const Mat& img_2_orig,
						   const vector<KeyPoint>& correspImg1Pt);

#define pclp3(eigenv3f) pcl::PointXYZ(eigenv3f.x(),eigenv3f.y(),eigenv3f.z())

pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,cloud_no_floor,orig_cloud;

void viewerOneOff (pcl::visualization::PCLVisualizer& viewer)
{
//	viewer.setBackgroundColor(255,255,255); //white background
//	viewer.removeCoordinateSystem();	//remove the axes
}

void SORFilter() {
	
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
	
	std::cerr << "Cloud before SOR filtering: " << cloud->width * cloud->height << " data points" << std::endl;
	

	// Create the filtering object
	pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
	sor.setInputCloud (cloud);
	sor.setMeanK (50);
	sor.setStddevMulThresh (1.0);
	sor.filter (*cloud_filtered);
	
	std::cerr << "Cloud after SOR filtering: " << cloud_filtered->width * cloud_filtered->height << " data points " << std::endl;
	
	copyPointCloud(*cloud_filtered,*cloud);
	copyPointCloud(*cloud,*orig_cloud);
	
	std::cerr << "PointCloud before VoxelGrid filtering: " << cloud->width * cloud->height << " data points (" << pcl::getFieldsList (*cloud) << ")."<<std::endl;
	
	cloud_filtered.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
	
	// Create the filtering object
	pcl::VoxelGrid<pcl::PointXYZRGB> vgrid;
	vgrid.setInputCloud (cloud);
	vgrid.setLeafSize (0.1f, 0.1f, 0.1f);
	vgrid.filter (*cloud_filtered);
	
	std::cerr << "PointCloud after VoxelGrid filtering: " << cloud_filtered->width * cloud_filtered->height << " data points (" << pcl::getFieldsList (*cloud_filtered) << ")."<<std::endl;	
	
	copyPointCloud(*cloud_filtered,*cloud);
}	

void viewerThread (pcl::visualization::PCLVisualizer& viewer)
{

}

void RunVisualization(const vector<cv::Point3d>& pointcloud,
					  const std::vector<cv::Vec3b>& pointcloud_RGB,
					  const Mat& img_1_orig, 
					  const Mat& img_2_orig,
					  const vector<KeyPoint>& correspImg1Pt) {
	cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
	orig_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);

	PopulatePCLPointCloud(pointcloud,pointcloud_RGB,img_1_orig,img_2_orig,correspImg1Pt);
//	SORFilter(); // apply a statistical outlier removal, will clean up the cloud
	copyPointCloud(*cloud,*orig_cloud);

	pcl::visualization::CloudViewer viewer("Cloud Viewer");

	//blocks until the cloud is actually rendered
	viewer.showCloud(orig_cloud,"orig");

	//This will only get called once
	viewer.runOnVisualizationThreadOnce (viewerOneOff);

	//This will get called once per visualization iteration
	viewer.runOnVisualizationThread (viewerThread);
	while (!viewer.wasStopped ())
	{
		;
	}
}	

void PopulatePCLPointCloud(const vector<Point3d>& pointcloud, 
						   const std::vector<cv::Vec3b>& pointcloud_RGB,
						   const Mat& img_1_orig, 
						   const Mat& img_2_orig,
						   const vector<KeyPoint>& correspImg1Pt)
	//Populate point cloud
{
	cout << "Creating point cloud...";
	double t = getTickCount();
	Mat_<Vec3b> img1_v3b,img2_v3b;
	if (!img_1_orig.empty() && !img_2_orig.empty()) {
		img1_v3b = Mat_<Vec3b>(img_1_orig);
		img2_v3b = Mat_<Vec3b>(img_2_orig);
	}
	for (unsigned int i=0; i<pointcloud.size(); i++) {
		Vec3b rgbv(255,255,255);
		if(!img_1_orig.empty()) {
			Point p = correspImg1Pt[i].pt;
			//		Point p1 = pt_set2[i];
			rgbv = img1_v3b(p.y,p.x); //(img1_v3b(p.y,p.x) + img2_v3b(p1.y,p1.x)) * 0.5;
		} else if (pointcloud_RGB.size()>0) {
			rgbv = pointcloud_RGB[i];
		}

		
		if (pointcloud[i].x != pointcloud[i].x || isnan(pointcloud[i].x) ||
			pointcloud[i].y != pointcloud[i].y || isnan(pointcloud[i].y) || 
			pointcloud[i].z != pointcloud[i].z || isnan(pointcloud[i].z) ||
			fabsf(pointcloud[i].x) > 10.0 || 
			fabsf(pointcloud[i].y) > 10.0 || 
			fabsf(pointcloud[i].z) > 10.0) {
			continue;
		}
		
		pcl::PointXYZRGB pclp;
		pclp.x = pointcloud[i].x;
		pclp.y = pointcloud[i].y;
		pclp.z = pointcloud[i].z;
		uint32_t rgb = ((uint32_t)rgbv[2] << 16 | (uint32_t)rgbv[1] << 8 | (uint32_t)rgbv[0]);
		pclp.rgb = *reinterpret_cast<float*>(&rgb);
		cloud->push_back(pclp);
	}
	cloud->width = (uint32_t) cloud->points.size();
	cloud->height = 1;
	t = ((double)getTickCount() - t)/getTickFrequency();
	cout << "Done. (" << t <<"s)"<< endl;
	pcl::PLYWriter pw;
	pw.write("pointcloud.ply",*cloud);
}

