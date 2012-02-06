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
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include <pcl/sample_consensus/sac_model_normal_plane.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>

#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;
 
void PopulatePCLPointCloud(const vector<Point3d>& pointcloud, 
						   const Mat& img_1_orig, 
						   const Mat& img_2_orig,
						   const vector<Point>& correspImg1Pt);
void FindNormalsMLS();
void FindFloorPlaneRANSAC();

#define pclp3(eigenv3f) pcl::PointXYZ(eigenv3f.x(),eigenv3f.y(),eigenv3f.z())

pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,cloud_no_floor,orig_cloud;
pcl::PointCloud<pcl::PointXYZ>::Ptr floorcloud;
pcl::RandomSampleConsensus<pcl::PointXYZRGB>::Ptr ransac;
Eigen::VectorXf coeffs[2];
pcl::IndicesPtr inliers;
pcl::PointCloud<pcl::Normal>::Ptr mls_normals;

void viewerOneOff (pcl::visualization::PCLVisualizer& viewer)
{
	viewer.setBackgroundColor(255,255,255);
	viewer.removeCoordinateSystem();
	
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(floorcloud, 0, 255, 0); 
	viewer.addPointCloud(floorcloud,single_color,"floor");
	
	//	cloud_no_floor.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
	//	pcl::ExtractIndices<pcl::PointXYZRGB> extract;
	//	extract.setInputCloud (cloud);
	//    extract.setIndices (inliers);
	//    extract.setNegative (true);
	//    extract.filter (*cloud_no_floor);
	//
	//	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> single_color1(cloud_no_floor, 255, 0, 0); 
	//	viewer.addPointCloud(cloud_no_floor,single_color1,"cloud");
	//	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,1.0,0,0,"floor");
	
	//	viewer.addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal >(cloud,mls_normals,10,0.08);
	
	using namespace Eigen;
	
	for (int c=1; c<2; c++) {
		VectorXf coeffs_ = coeffs[c];
		Vector3f n(coeffs_[0],coeffs_[1],coeffs_[2]);
		Vector3f onplane1 = n.cross(Vector3f::UnitX()).normalized();
		Vector3f onplane2 = n.cross(onplane1).normalized();		
		
		//draw a grid for the floor plane
		for(int i=0;i<26;i++) {
			Vector3f p1 = n * -coeffs_[3] + onplane1 * 2.0 * (double)(i-9)/20.0;
			Vector3f p2 = p1 + onplane2 * 2.5;
			stringstream ss; ss<<"line"<<c<<i;
			viewer.addLine<pcl::PointXYZ,pcl::PointXYZ>(pclp3(p1),pclp3(p2),1.0,1.0*c,0,ss.str());
			
			p1 = n * -coeffs_[3] + onplane1 * 2.0 * (double)(-9)/20.0 + onplane2 * 2.0 * (double)(i)/20.0;
			p2 = p1 + onplane1 * 2.5;
			ss<<"opp";
			viewer.addLine<pcl::PointXYZ,pcl::PointXYZ>(pclp3(p1),pclp3(p2),1.0,1.0*c,0,ss.str());		
		}
	}
}

void SORFilter() {
	orig_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
	
	std::cerr << "Cloud before SOR filtering: " << cloud->width * cloud->height << " data points" << std::endl;
	
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);

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
//	viewer.setCameraPosition(
}

void RunVisualization(const vector<cv::Point3d>& pointcloud,
					  const Mat& img_1_orig, 
					  const Mat& img_2_orig,
					  const vector<Point>& correspImg1Pt) {
	cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
	//  pcl::io::loadPCDFile ("output.pcd", *cloud);
    
	PopulatePCLPointCloud(pointcloud,img_1_orig,img_2_orig,correspImg1Pt);
	SORFilter();
	FindNormalsMLS();
	FindFloorPlaneRANSAC();
	
	//	pcl::PointCloud<pcl::PointXYZRGB>::Ptr final (new pcl::PointCloud<pcl::PointXYZRGB>);
	//	pcl::copyPointCloud<pcl::PointXYZRGB>(*cloud, inliers, *final);
	
    pcl::visualization::CloudViewer viewer("Cloud Viewer");
    
    //blocks until the cloud is actually rendered
    viewer.showCloud(orig_cloud,"orig");
	//	viewer.showCloud(final);
    
    //use the following functions to get access to the underlying more advanced/powerful
    //PCLVisualizer
    
    //This will only get called once
    viewer.runOnVisualizationThreadOnce (viewerOneOff);
    
    //This will get called once per visualization iteration
	viewer.runOnVisualizationThread (viewerThread);
    while (!viewer.wasStopped ())
    {
		//you can also do cool processing here
		//FIXME: Note that this is running in a separate thread from viewerPsycho
		//and you should guard against race conditions yourself...
		//		user_data++;
    }
}	

void PopulatePCLPointCloud(const vector<Point3d>& pointcloud, 
						   const Mat& img_1_orig, 
						   const Mat& img_2_orig,
						   const vector<Point>& correspImg1Pt)
	//Populate point cloud
{
	cout << "Creating point cloud...";
	double t = getTickCount();
	Mat_<Vec3b> img1_v3b(img_1_orig),img2_v3b(img_2_orig);
	for (unsigned int i=0; i<pointcloud.size(); i++) {
		Point p = correspImg1Pt[i];
		//		Point p1 = pt_set2[i];
		Vec3b rgbv = img1_v3b(p.y,p.x); //(img1_v3b(p.y,p.x) + img2_v3b(p1.y,p1.x)) * 0.5;
		
		pcl::PointXYZRGB pclp;
		pclp.x = pointcloud[i].x;
		pclp.y = pointcloud[i].y;
		pclp.z = pointcloud[i].z;
		uint32_t rgb = ((uint32_t)rgbv[2] << 16 | (uint32_t)rgbv[1] << 8 | (uint32_t)rgbv[0]);
		pclp.rgb = *reinterpret_cast<float*>(&rgb);
		cloud->push_back(pclp);
	}
	t = ((double)getTickCount() - t)/getTickFrequency();
	cout << "Done. (" << t <<"s)"<< endl;
	pcl::PLYWriter pw;
	pw.write("pointcloud.ply",*cloud);
}

void FindNormalsMLS()
//find normals using MLS
{
	double t = getTickCount();
	cout << "MLS...";
	
	mls_normals.reset(new pcl::PointCloud<pcl::Normal> ());
	
	// Create a KD-Tree
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
	
	// Output has the same type as the input one, it will be only smoothed
	pcl::PointCloud<pcl::PointXYZRGB> mls_points;
	
	// Init object (second point type is for the normals, even if unused)
	pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::Normal> mls;
	
	// Optionally, a pointer to a cloud can be provided, to be set by MLS
	mls.setOutputNormals (mls_normals);
	
	// Set parameters
	mls.setInputCloud (cloud);
	mls.setPolynomialFit (true);
	mls.setSearchMethod (tree);
	mls.setSearchRadius (0.16);
	
	// Reconstruct
	mls.reconstruct (mls_points);
	pcl::copyPointCloud<pcl::PointXYZRGB,pcl::PointXYZRGB>(mls_points,*cloud);
	
	t = ((double)getTickCount() - t)/getTickFrequency();
	cout << "Done. (" << t <<"s)"<< endl;
}

void FindFloorPlaneRANSAC()
{	
	double t = getTickCount();
	cout << "RANSAC...";
	/*
	 pcl::SampleConsensusModelPlane<pcl::PointXYZRGB>::Ptr model_p (new pcl::SampleConsensusModelPlane<pcl::PointXYZRGB> (cloud));
	 */
	pcl::SampleConsensusModelNormalPlane<pcl::PointXYZRGB,pcl::Normal>::Ptr model_p(
					new pcl::SampleConsensusModelNormalPlane<pcl::PointXYZRGB,pcl::Normal>(cloud));
	//	model_p->setInputCloud(cloud);
	model_p->setInputNormals(mls_normals);
	model_p->setNormalDistanceWeight(0.75);
	
	inliers.reset(new vector<int>);
	
	ransac.reset(new pcl::RandomSampleConsensus<pcl::PointXYZRGB>(model_p));
	ransac->setDistanceThreshold (.1);
	ransac->computeModel();	
	ransac->getInliers(*inliers);
	
	t = ((double)getTickCount() - t)/getTickFrequency();
	cout << "Done. (" << t <<"s)"<< endl;
	
	ransac->getModelCoefficients(coeffs[0]);
	model_p->optimizeModelCoefficients(*inliers,coeffs[0],coeffs[1]);
	
	floorcloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::copyPointCloud(*cloud,*inliers,*floorcloud);

}