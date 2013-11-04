/*
 *  cloud_viewer.cpp
 *  EyeRingOpenCV
 *
 *  Created by Roy Shilkrot on 12/17/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */
#include <pcl/common/common.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/io.h>
#include <pcl/io/file_io.h>
#include <pcl/io/pcd_io.h>

pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;

void 
viewerOneOff (pcl::visualization::PCLVisualizer& viewer)
{
	viewer.setBackgroundColor(255,255,255);
//    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
//	viewer.addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
}


int main ()
{
	cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPCDFile ("/Users/royshilkrot/Downloads/EyeRing-OpenCV/output.pcd", *cloud);
    
    pcl::visualization::CloudViewer viewer("Cloud Viewer");
    
    //blocks until the cloud is actually rendered
    viewer.showCloud(cloud);
    
    //use the following functions to get access to the underlying more advanced/powerful
    //PCLVisualizer
    
    //This will only get called once
    viewer.runOnVisualizationThreadOnce (viewerOneOff);
    
    //This will get called once per visualization iteration
//    viewer.runOnVisualizationThread (viewerPsycho);
    while (!viewer.wasStopped ())
    {
		//you can also do cool processing here
		//FIXME: Note that this is running in a separate thread from viewerPsycho
		//and you should guard against race conditions yourself...
//		user_data++;
    }
    return 0;
}