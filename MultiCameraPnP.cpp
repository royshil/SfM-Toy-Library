/*
 *  MultiCameraPnP.cpp
 *  SfMToyExample
 *
 *  Created by Roy Shilkrot on 4/27/12.
 *  Copyright 2012 MIT. All rights reserved.
 *
 */
#define USE_PROFILING

#include "MultiCameraPnP.h"
#include "BundleAdjuster.h"

#include <list>
#include <set>

using namespace std;

#include <opencv2/gpu/gpu.hpp>

bool sort_by_first(pair<int,pair<int,int> > a, pair<int,pair<int,int> > b) { return a.first < b.first; }

/**
 * Get an initial 3D point cloud from 2 views only
 */
void MultiCameraPnP::GetBaseLineTriangulation() {
	std::cout << "=========================== Baseline triangulation ===========================\n";

	cv::Matx34d P(1,0,0,0,
				  0,1,0,0,
				  0,0,1,0),
				P1(1,0,0,0,
				   0,1,0,0,
				   0,0,1,0);
	
	std::vector<CloudPoint> tmp_pcloud;

	//sort pairwise matches to find the highest match [Snavely07 4.2]
	list<pair<int,pair<int,int> > > matches_sizes;
	for(std::map<std::pair<int,int> ,std::vector<cv::DMatch> >::iterator i = matches_matrix.begin(); i != matches_matrix.end(); ++i) {
		matches_sizes.push_back(make_pair((*i).second.size(),(*i).first));
	}
	matches_sizes.sort(sort_by_first);

	//Reconstruct from two views
	bool goodF = false;
	int highest_pair = 0;
	m_first_view = m_second_view = 0;
	//reverse iterate by number of matches
	for(list<pair<int,pair<int,int> > >::reverse_iterator highest_pair = matches_sizes.rbegin(); 
		highest_pair != matches_sizes.rend() && !goodF; 
		++highest_pair) 
	{
		m_second_view = (*highest_pair).second.second;
		m_first_view  = (*highest_pair).second.first;

		std::cout << " -------- " << imgs_names[m_first_view] << " and " << imgs_names[m_second_view] << " -------- " <<std::endl;
		//what if reconstrcution of first two views is bad? fallback to another pair
		//See if the Fundamental Matrix between these two views is good
		goodF = FindCameraMatrices(K, Kinv, distortion_coeff,
			imgpts[m_first_view], 
			imgpts[m_second_view], 
			imgpts_good[m_first_view],
			imgpts_good[m_second_view], 
			P, 
			P1,
			matches_matrix[std::make_pair(m_first_view,m_second_view)],
			tmp_pcloud
#ifdef __SFM__DEBUG__
			,imgs[m_first_view],imgs[m_second_view]
#endif
		);
		if (!goodF) {
			m_second_view++; //go to the next view...
		} else {
			vector<CloudPoint> new_triangulated;
			vector<int> add_to_cloud;

			Pmats[m_first_view] = P;
			Pmats[m_second_view] = P1;

			bool good_triangulation = TriangulatePointsBetweenViews(m_second_view,m_first_view,new_triangulated,add_to_cloud);
			if(!good_triangulation) {
				std::cout << "triangulation failed" << std::endl;
				goodF = false;
				Pmats[m_first_view] = 0;
				Pmats[m_second_view] = 0;
				m_second_view++;
			} else {
				std::cout << "before triangulation: " << pcloud.size();
				for (unsigned int j=0; j<add_to_cloud.size(); j++) {
					if(add_to_cloud[j] == 1)
						pcloud.push_back(new_triangulated[j]);
				}
				std::cout << " after " << pcloud.size() << std::endl;
			}				
		}
		if (!goodF) {
			m_first_view++;
		}
	}
		
	if (!goodF) {
		cerr << "Cannot find a good pair of images to obtain a baseline triangulation" << endl;
		exit(0);
	}
	
	cout << "Taking baseline from " << imgs_names[m_first_view] << " and " << imgs_names[m_second_view] << endl;
	
//	double reproj_error;
//	{
//		std::vector<cv::KeyPoint> pt_set1,pt_set2;
//		
//		std::vector<cv::DMatch> matches = matches_matrix[std::make_pair(m_first_view,m_second_view)];
//
//		GetAlignedPointsFromMatch(imgpts[m_first_view],imgpts[m_second_view],matches,pt_set1,pt_set2);
//		
//		pcloud.clear();
//		reproj_error = TriangulatePoints(pt_set1, 
//										 pt_set2, 
//										 Kinv, 
//										 distortion_coeff,
//										 Pmats[m_first_view], 
//										 Pmats[m_second_view], 
//										 pcloud, 
//										 correspImg1Pt);
//		
//		for (unsigned int i=0; i<pcloud.size(); i++) {
//			pcloud[i].imgpt_for_img = std::vector<int>(imgs.size(),-1);
//			//matches[i] corresponds to pointcloud[i]
//			pcloud[i].imgpt_for_img[m_first_view] = matches[i].queryIdx;
//			pcloud[i].imgpt_for_img[m_second_view] = matches[i].trainIdx;
//		}
//	}
//	std::cout << "triangulation reproj error " << reproj_error << std::endl;
}

void MultiCameraPnP::Find2D3DCorrespondences(int working_view, 
	std::vector<cv::Point3f>& ppcloud, 
	std::vector<cv::Point2f>& imgPoints) 
{
	ppcloud.clear(); imgPoints.clear();

	vector<int> pcloud_status(pcloud.size(),0);
	for (int old_view = imgs.size()-1; old_view >= 0; old_view--)
	{
		if(old_view == working_view) continue;

		//check for matches_from_old_to_working between i'th frame and <old_view>'th frame (and thus the current cloud)
		std::vector<cv::DMatch> matches_from_old_to_working = matches_matrix[std::make_pair(old_view,working_view)];

		for (unsigned int match_from_old_view=0; match_from_old_view < matches_from_old_to_working.size(); match_from_old_view++) {
			// the index of the matching point in <old_view>
			int idx_in_old_view = matches_from_old_to_working[match_from_old_view].queryIdx;

			//scan the existing cloud (pcloud) to see if this point from <old_view> exists
			for (unsigned int pcldp=0; pcldp<pcloud.size(); pcldp++) {
				// see if corresponding point was found in cloud
				if (idx_in_old_view == pcloud[pcldp].imgpt_for_img[old_view] && pcloud_status[pcldp] == 0) //prevent duplicates
				{
					//3d point in cloud
					ppcloud.push_back(pcloud[pcldp].pt);
					//2d point in image i
					imgPoints.push_back(imgpts[working_view][matches_from_old_to_working[match_from_old_view].trainIdx].pt);

					pcloud_status[pcldp] = 1;
					break;
				}
			}
		}
	}
	cout << "found " << ppcloud.size() << " 3d-2d point correspondences"<<endl;
}

bool MultiCameraPnP::FindPoseEstimation(
	int working_view,
	cv::Mat_<double>& rvec,
	cv::Mat_<double>& t,
	cv::Mat_<double>& R
	) 
{

	std::vector<cv::Point3f> ppcloud;
	std::vector<cv::Point2f> imgPoints;
	CV_PROFILE("Find2D3DCorrespondences",Find2D3DCorrespondences(working_view,ppcloud,imgPoints);)

	if(ppcloud.size() <= 7 || imgPoints.size() <= 7 || ppcloud.size() != imgPoints.size()) { 
		//something went wrong aligning 3D to 2D points..
		cerr << "couldn't find [enough] corresponding cloud points... (only " << ppcloud.size() << ")" <<endl;
		return false;
	}

	if(!use_gpu) {
		//use CPU
		vector<int> inliers;
		CV_PROFILE("solvePnPRansac",cv::solvePnPRansac(ppcloud, imgPoints, K, distortion_coeff, rvec, t, true, 1000, 10.0, 0.25 * (double)(imgPoints.size()), inliers, CV_EPNP);)
		//CV_PROFILE("solvePnP",cv::solvePnP(ppcloud, imgPoints, K, distortion_coeff, rvec, t, true, CV_EPNP);)

		vector<cv::Point2f> projected3D;
		cv::projectPoints(ppcloud, rvec, t, K, distortion_coeff, projected3D);
		if(inliers.size()==0) { //get inliers
			for(int i=0;i<projected3D.size();i++) {
				if(norm(projected3D[i]-imgPoints[i]) < 20.0)
					inliers.push_back(i);
			}
		}

		cv::Mat reprojected; imgs_orig[working_view].copyTo(reprojected);
		for (int ppt=0; ppt<inliers.size(); ppt++) {
			cv::line(reprojected,imgPoints[inliers[ppt]],projected3D[inliers[ppt]],cv::Scalar(0,0,255),1);
		}
		//for(int ppt=0;ppt<imgPoints.size();ppt++) {
		//	cv::line(reprojected,imgPoints[ppt],projected3D[ppt],cv::Scalar(0,0,255),1);
		//}
		for(int ppt=0;ppt<imgPoints.size();ppt++) {
			cv::circle(reprojected, imgPoints[ppt], 2, cv::Scalar(255,0,0), CV_FILLED);
			cv::circle(reprojected, projected3D[ppt], 2, cv::Scalar(0,255,0), CV_FILLED);			
		}
		for (int ppt=0; ppt<inliers.size(); ppt++) {
			cv::circle(reprojected, imgPoints[inliers[ppt]], 2, cv::Scalar(255,255,0), CV_FILLED);
		}
		stringstream ss; ss << "inliers " << inliers.size() << " / " << projected3D.size();
		putText(reprojected, ss.str(), cv::Point(5,20), CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0,255,255), 2);

		cv::imshow("__tmp", reprojected);
		cv::waitKey(0);
		cv::destroyWindow("__tmp");

		cv::Rodrigues(rvec, R);
		visualizerShowCamera(R,t,0,255,0,0.1);

		if(inliers.size() < (double)(imgPoints.size())/5.0) {
			cerr << "not enough inliers to consider a good pose ("<<inliers.size()<<"/"<<imgPoints.size()<<")"<< endl;
			return false;
		}
	} else {
		//use GPU ransac
		//make sure datatstructures are cv::gpu compatible
		cv::Mat ppcloud_m(ppcloud); ppcloud_m = ppcloud_m.t();
		cv::Mat imgPoints_m(imgPoints); imgPoints_m = imgPoints_m.t();
		cv::Mat rvec_,t_;

		cv::gpu::solvePnPRansac(ppcloud_m,imgPoints_m,K_32f,distcoeff_32f,rvec_,t_,false);

		rvec_.convertTo(rvec,CV_64FC1);
		t_.convertTo(t,CV_64FC1);
	}

	if(cv::norm(t) > 200.0) {
		// this is bad...
		cerr << "estimated camera movement is too big, skip this camera\r\n";
		return false;
	}

	cv::Rodrigues(rvec, R);
	if(!CheckCoherentRotation(R)) {
		cerr << "rotation is incoherent. we should try a different base view..." << endl;
		return false;
	}

	std::cout << "found t = " << t << "\nR = \n"<<R<<std::endl;
}

bool MultiCameraPnP::TriangulatePointsBetweenViews(
	int working_view, 
	int older_view,
	vector<struct CloudPoint>& new_triangulated,
	vector<int>& add_to_cloud
	) 
{
	//get the left camera matrix
	//TODO: potential bug - the P mat for <view> may not exist? or does it...
	cv::Matx34d P = Pmats[older_view];
	cv::Matx34d P1 = Pmats[working_view];

	//prune the match between i and <view> using the Fundamental matrix to prune
	GetFundamentalMat( imgpts[older_view], 
		imgpts[working_view], 
		imgpts_good[older_view],
		imgpts_good[working_view], 
		matches_matrix[std::make_pair(older_view,working_view)]
#ifdef __SFM__DEBUG__
	,imgs_orig[older_view], imgs_orig[working_view]
#endif
	);			

	std::vector<cv::KeyPoint> pt_set1,pt_set2;
	std::vector<cv::DMatch> matches = matches_matrix[std::make_pair(older_view,working_view)];
	GetAlignedPointsFromMatch(imgpts[older_view],imgpts[working_view],matches,pt_set1,pt_set2);

	unsigned int start_i = pcloud.size();


	//adding more triangulated points to general cloud
	double reproj_error = TriangulatePoints(pt_set1, pt_set2, Kinv, distortion_coeff, P, P1, new_triangulated, correspImg1Pt);
	std::cout << "triangulation reproj error " << reproj_error << std::endl;

	if(reproj_error > 100.0) {
		// somethign went awry, delete those triangulated points
		//				pcloud.resize(start_i);
		cerr << "reprojection error too high, don't include these points."<<endl;
		return false;
	}

	add_to_cloud.clear();
	add_to_cloud.resize(new_triangulated.size(),1);
	int found_other_views_count = 0;

	//scan new triangulated points, if they were already triangulated before - strengthen cloud
	//#pragma omp parallel for num_threads(1)
	for (int j = 0; j<matches.size(); j++) {
		new_triangulated[j].imgpt_for_img = std::vector<int>(imgs.size(),-1);

		//matches[j] corresponds to new_triangulated[j]
		//matches[j].queryIdx = point in <m_second_view>
		//matches[j].trainIdx = point in <working_view>
		new_triangulated[j].imgpt_for_img[older_view] = matches[j].queryIdx;	//2D reference to <view>
		new_triangulated[j].imgpt_for_img[working_view] = matches[j].trainIdx;		//2D reference to <i>
		bool found_in_other_view = false;
		for (unsigned int view_ = 0; view_ < working_view; view_++) {
			if(view_ != older_view) {
				//Look for points in <view_> that match to points in <working_view>
				std::vector<cv::DMatch> submatches = matches_matrix[std::make_pair(view_,working_view)];
				for (unsigned int ii = 0; ii < submatches.size(); ii++) {
					if (submatches[ii].trainIdx == matches[j].trainIdx &&
						!found_in_other_view) 
					{
						//Point was already found in <view_> - strengthen it in the known cloud, if it exists there

						//cout << "2d pt " << submatches[ii].queryIdx << " in img " << view_ << " matched 2d pt " << submatches[ii].trainIdx << " in img " << i << endl;
						for (unsigned int pt3d=0; pt3d<pcloud.size(); pt3d++) {
							if (pcloud[pt3d].imgpt_for_img[view_] == submatches[ii].queryIdx) 
							{
								//cout << "3d point "<<pt3d<<" in cloud, referenced 2d pt " << submatches[ii].queryIdx << " in view " << view_ << endl;
#pragma omp critical 
								{
									pcloud[pt3d].imgpt_for_img[working_view] = matches[j].trainIdx;
									pcloud[pt3d].imgpt_for_img[older_view] = matches[j].queryIdx;
									found_in_other_view = true;
									add_to_cloud[j] = 0;
								}
							}
						}
					}
				}
			}
		}
#pragma omp critical
		{
			if (found_in_other_view) {
				found_other_views_count++;
			} else {
				add_to_cloud[j] = 1;
			}
		}
	}
	std::cout << found_other_views_count << "/" << matches.size() << " points were found in other views, adding " << cv::countNonZero(add_to_cloud) << " new\n";
	return true;
}

void MultiCameraPnP::GetRGBForPointCloud(
	const std::vector<struct CloudPoint>& pcloud,
	std::vector<cv::Vec3b>& RGBforCloud
	) 
{
	for (unsigned int i=0; i<pcloud.size(); i++) {
		//pointcloud_beforeBA.push_back(pcloud[i]);
		unsigned int good_view = 0;
		for(; good_view < imgs_orig.size(); good_view++) {
			if(pcloud[i].imgpt_for_img[good_view] != -1) {
				int pt_idx = pcloud[i].imgpt_for_img[good_view];
				if(pt_idx >= imgpts[good_view].size()) {
					cerr << "BUG: point id:" << pt_idx << " should not exist for img #" << good_view << " which has only " << imgpts[good_view].size() << endl;
					continue;
				}
				cv::Point _pt = imgpts[good_view][pt_idx].pt;
				assert(good_view < imgs_orig.size() && _pt.x < imgs_orig[good_view].cols && _pt.y < imgs_orig[good_view].rows);
				RGBforCloud.push_back(imgs_orig[good_view].at<cv::Vec3b>(_pt));
				break;
			}
		}
		if(good_view == imgs.size()) //nothing found.. put red dot
			RGBforCloud.push_back(cv::Vec3b(255,0,0));
	}
}

void MultiCameraPnP::AdjustCurrentBundle() {
	cout << "======================== Bundle Adjustment ==========================\n";

	pointcloud_beforeBA = pcloud;
	GetRGBForPointCloud(pointcloud_beforeBA,pointCloudRGB_beforeBA);
	
	BundleAdjuster BA;
	BA.adjustBundle(pcloud,cam_matrix,imgpts,Pmats);
//	K = cam_matrix;
//	Kinv = K.inv();
	
	pointcloud = pcloud;
	GetRGBForPointCloud(pointcloud,pointCloudRGB);
}	

void MultiCameraPnP::RecoverDepthFromImages() {
	if(!features_matched) 
		OnlyMatchFeatures();
	
	std::cout << "======================================================================\n";
	std::cout << "======================== Depth Recovery Start ========================\n";
	std::cout << "======================================================================\n";
	
	GetBaseLineTriangulation();
	//AdjustCurrentBundle();
	//return;
	
	cv::Matx34d P1 = Pmats[m_second_view];
	cv::Mat_<double> t = (cv::Mat_<double>(1,3) << P1(0,3), P1(1,3), P1(2,3));
	cv::Mat_<double> R = (cv::Mat_<double>(3,3) << P1(0,0), P1(0,1), P1(0,2), 
												   P1(1,0), P1(1,1), P1(1,2), 
												   P1(2,0), P1(2,1), P1(2,2));
	cv::Mat_<double> rvec(1,3); Rodrigues(R, rvec);
	
	set<int> done_views;
	done_views.insert(m_first_view);
	done_views.insert(m_second_view);

	//loop images to incrementally recover more cameras 
	//for (unsigned int i=0; i < imgs.size(); i++) 
	while (done_views.size() != imgs.size())
	{
		//find image with highest 2d-3d correspondance [Snavely07 4.2]
		unsigned int max_2d3d_view = -1, max_2d3d_count = 0;
		for (unsigned int i=0; i < imgs.size(); i++) {
			if(done_views.find(i) != done_views.end()) continue; //already done with this view

			vector<cv::Point3f> tmp3d; vector<cv::Point2f> tmp2d;
			Find2D3DCorrespondences(i,tmp3d,tmp2d);
			if(tmp3d.size() > max_2d3d_count) {
				max_2d3d_count = tmp3d.size();
				max_2d3d_view = i;
			}
		}
		int i = max_2d3d_view;

		std::cout << "-------------------------- " << imgs_names[i] << " --------------------------\n";
		done_views.insert(i); // don't repeat it for now

		bool pose_estimated = FindPoseEstimation(i,rvec,t,R);
		if(!pose_estimated)
			continue;

		//store estimated pose	
		Pmats[i] = cv::Matx34d	(R(0,0),R(0,1),R(0,2),t(0),
								 R(1,0),R(1,1),R(1,2),t(1),
								 R(2,0),R(2,1),R(2,2),t(2));
		
		// start triangulating with previous views
		for (int view = i-1; view >= 0; view--) 
		{
			cout << " -> " << imgs_names[view] << endl;
			
			vector<CloudPoint> new_triangulated;
			vector<int> add_to_cloud;
			bool good_triangulation = TriangulatePointsBetweenViews(i,view,new_triangulated,add_to_cloud);
			if(!good_triangulation) continue;

			std::cout << "before triangulation: " << pcloud.size();
			for (int j=0; j<add_to_cloud.size(); j++) {
				if(add_to_cloud[j] == 1)
					pcloud.push_back(new_triangulated[j]);
			}
			std::cout << " after " << pcloud.size() << std::endl;

			//break;
		}
	}
	
	AdjustCurrentBundle();

	cout << "======================================================================\n";
	cout << "========================= Depth Recovery DONE ========================\n";
	cout << "======================================================================\n";
}
