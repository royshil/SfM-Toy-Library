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

#include <opencv2/core/core.hpp>

using namespace std;
 
void PopulatePCLPointCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& mycloud,
						   const vector<cv::Point3d>& pointcloud, 
						   const std::vector<cv::Vec3b>& pointcloud_RGB,
						   bool write_to_file = false
						   );

#define pclp3(eigenv3f) pcl::PointXYZ(eigenv3f.x(),eigenv3f.y(),eigenv3f.z())

pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,cloud1,cloud_no_floor,orig_cloud;

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

bool show_cloud = false;
bool sor_applied = false;
bool show_cloud_A = true;

void keyboardEventOccurred (const pcl::visualization::KeyboardEvent& event_,
                            void* viewer_void)
{
	pcl::visualization::CloudViewer* viewer = static_cast<pcl::visualization::CloudViewer *> (viewer_void);
	cout << "event_.getKeySym () = " << event_.getKeySym () << " event_.keyDown () " << event_.keyDown () << endl;
	if ((event_.getKeySym () == "s" || event_.getKeySym () == "S") && event_.keyDown ())
	{
		cout << "s clicked" << endl;
		
		cloud->clear();
		copyPointCloud(*orig_cloud,*cloud);
		if (!sor_applied) {
			SORFilter();
			sor_applied = true;
		} else {
			sor_applied = false;
		}

		show_cloud = true;
	}
	if ((event_.getKeySym ().compare("1") == 0)
#ifndef WIN32
		&& event_.keyDown ()
#endif
		) 
	{
		show_cloud_A = true;
		show_cloud = true;
	}
	if ((event_.getKeySym ().compare("2") == 0)
#ifndef WIN32
		&& event_.keyDown ()
#endif
		) 
	{
		show_cloud_A = false;
		show_cloud = true;
	}
}

void RunVisualization(const vector<cv::Point3d>& pointcloud,
					  const vector<cv::Vec3b>& pointcloud_RGB,
					  const vector<cv::Point3d>& pointcloud1,
					  const vector<cv::Vec3b>& pointcloud1_RGB) 
{
	cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
	cloud1.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
	orig_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    
	PopulatePCLPointCloud(cloud,pointcloud,pointcloud_RGB);
	PopulatePCLPointCloud(cloud1,pointcloud1,pointcloud1_RGB);
	copyPointCloud(*cloud,*orig_cloud);
	
    pcl::visualization::CloudViewer viewer("Cloud Viewer");
    
    //blocks until the cloud is actually rendered
    viewer.showCloud(orig_cloud,"orig");
	
	viewer.registerKeyboardCallback (keyboardEventOccurred, (void*)&viewer);
	
    while (!viewer.wasStopped ())
    {
		if (show_cloud) {
			cout << "Show cloud\n";
			if(show_cloud_A)
				viewer.showCloud(cloud,"orig");
			else
				viewer.showCloud(cloud1,"orig");
			show_cloud = false;
		}
    }
}	

void PopulatePCLPointCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& mycloud,
						   const vector<cv::Point3d>& pointcloud, 
						   const std::vector<cv::Vec3b>& pointcloud_RGB,
						   bool write_to_file
						   )
	//Populate point cloud
{
	cout << "Creating point cloud...";
	double t = cv::getTickCount();

	for (unsigned int i=0; i<pointcloud.size(); i++) {
		// get the RGB color value for the point
		cv::Vec3b rgbv(255,255,255);
		if (pointcloud_RGB.size() >= i) {
			rgbv = pointcloud_RGB[i];
		}

		// check for erroneous coordinates (NaN, Inf, etc.)
		if (pointcloud[i].x != pointcloud[i].x || 
			pointcloud[i].y != pointcloud[i].y || 
			pointcloud[i].z != pointcloud[i].z || 
#ifndef WIN32
			isnan(pointcloud[i].x) ||
			isnan(pointcloud[i].y) || 
			isnan(pointcloud[i].z) ||
#else
			_isnan(pointcloud[i].x) ||
			_isnan(pointcloud[i].y) || 
			_isnan(pointcloud[i].z) ||
#endif
			//fabsf(pointcloud[i].x) > 10.0 || 
			//fabsf(pointcloud[i].y) > 10.0 || 
			//fabsf(pointcloud[i].z) > 10.0
			false
			) 
		{
			continue;
		}
		
		pcl::PointXYZRGB pclp;
		
		// 3D coordinates
		pclp.x = pointcloud[i].x;
		pclp.y = pointcloud[i].y;
		pclp.z = pointcloud[i].z;
		
		// RGB color, needs to be represented as an integer
		uint32_t rgb = ((uint32_t)rgbv[2] << 16 | (uint32_t)rgbv[1] << 8 | (uint32_t)rgbv[0]);
		pclp.rgb = *reinterpret_cast<float*>(&rgb);
		
		mycloud->push_back(pclp);
	}
	
	mycloud->width = (uint32_t) mycloud->points.size(); // number of points
	mycloud->height = 1;								// a list, one row of data
	
	t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
	cout << "Done. (" << t <<"s)"<< endl;
	
	// write to file
	if (write_to_file) {
		//pcl::PLYWriter pw;
		//pw.write("pointcloud.ply",*mycloud);
		pcl::PCDWriter pw;
		pw.write("pointcloud.pcd",*mycloud);
	}
}
