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

#include "Common.h"

class MultiCameraPnP : public MultiCameraDistance {
public:
	MultiCameraPnP(const std::vector<cv::Mat>& imgs_, const std::vector<std::string>& imgs_names_):
	MultiCameraDistance(imgs_,imgs_names_) 
	{
	}
	
	virtual void RecoverDepthFromImages() {
		if(!features_matched) 
			OnlyMatchFeatures();
		
		std::cout << "======================================================================\n";
		std::cout << "======================================================================\n";

		cv::Matx34d P(1,0,0,0,
					  0,1,0,0,
					  0,0,1,0);
		
		std::vector<CloudPoint> tmp_pcloud;
		//Reconstruct from first two views
		FindCameraMatrices(K, Kinv, 
						   imgpts[0], 
						   imgpts[1], 
						   imgpts_good[0],
						   imgpts_good[1], 
						   P, 
						   Pmats[std::make_pair(0,1)],
						   matches_matrix[std::make_pair(0,1)],
						   tmp_pcloud
#ifdef __SFM__DEBUG__
						   ,imgs[0],imgs[1]
#endif
						   );
		//TODO: what if reconstrcution of first two views is bad? fallback to another pair
		
		
		//TODO: if the P1 matrix is far away from identity rotation - the solution is probably invalid...
		//so use an identity matrix?
//		cv::Matx34d P1 = Pmats[std::make_pair(0, 1)];
//		cv::Mat_<double> R = (cv::Mat_<double>(3,3) << P1(0,0),P1(0,1),P1(0,2),
//													  P1(1,0),P1(1,1),P1(1,2),
//													  P1(2,0),P1(2,1),P1(2,2));
//		if(!CheckCoherentRotation(R)) {
//			std::cout << " bad pair " << std::endl;
//			continue; //skip triangulation 
//		}
		
		//Get baseline triangulation
		std::vector<CloudPoint> pcloud;
		double reproj_error;
		{
			std::vector<cv::KeyPoint> pt_set1,pt_set2;

			std::vector<cv::DMatch> matches = matches_matrix[std::make_pair(0,1)];
			GetAlignedPointsFromMatch(imgpts[0],imgpts[1],matches,pt_set1,pt_set2);
			
			reproj_error = TriangulatePoints(pt_set1, 
											pt_set2, 
											Kinv, 
											P, 
											Pmats[std::make_pair(0,1)], 
											pcloud, 
											correspImg1Pt);

			std::cout << "pt_set1.size() " << pt_set1.size() << " pt_set2.size() " << pt_set2.size() << " matches.size() " << matches.size() << std::endl;

			for (unsigned int i=0; i<pcloud.size(); i++) {
				pcloud[i].idx_in_imgpts_for_img = std::vector<int>(imgs.size(),-1);
				//matches[i] corresponds to pointcloud[i]
				pcloud[i].idx_in_imgpts_for_img[0] = matches[i].queryIdx;
				pcloud[i].idx_in_imgpts_for_img[1] = matches[i].trainIdx;
			}
		}
		std::cout << "triangulation reproj error " << reproj_error << std::endl;
		
		//loop images to incrementally recover more cameras 
		for (unsigned int i=2; i < imgs.size(); i++) {
			
			//Update the match between i and 0
			FindCameraMatrices(K, Kinv, 
							   imgpts[0], 
							   imgpts[i], 
							   imgpts_good[0],
							   imgpts_good[i], 
							   P, 
							   Pmats[std::make_pair(0,i)],
							   matches_matrix[std::make_pair(0,i)],
							   tmp_pcloud
#ifdef __SFM__DEBUG__
							   ,imgs[0],imgs[i]
#endif
							   );

			//check for matches between i'th frame and 0'th frame (and thus the current cloud)
			std::vector<cv::Point3f> ppcloud;
			std::vector<cv::Point2f> imgPoints;
			std::vector<cv::DMatch> matches = matches_matrix[std::make_pair(0,i)];
			for (unsigned int pt_img0=0; pt_img0<matches.size(); pt_img0++) {
				int matches_img0_queryIdx = matches[pt_img0].queryIdx;
				for (unsigned int pcldp=0; pcldp<pcloud.size(); pcldp++) {
					if (matches_img0_queryIdx == pcloud[pcldp].idx_in_imgpts_for_img[0]) {
						//point in cloud
						ppcloud.push_back(pcloud[pcldp].pt);
						//point in image i
						imgPoints.push_back(imgpts[i][matches[pt_img0].trainIdx].pt);
						
						break;
					}
				}
			}
			
			cv::Mat_<double> t,rvec; cv::Mat_<double> distcoeff(1,4,0.0);
			
			cv::solvePnPRansac(ppcloud, imgPoints, K, distcoeff, rvec, t, false);
//			cv::solvePnP(ppcloud, imgPoints, K, distcoeff, rvec, t, false, CV_EPNP);
			
			cv::Mat_<double> R(3,3); Rodrigues(rvec, R);
			
			std::cout << "found t = " << t << "\nR = \n"<<R<<std::endl;
			
			cv::Matx34d P1 = cv::Matx34d(R(0,0),R(0,1),R(0,2),t(0),
										 R(1,0),R(1,1),R(1,2),t(1),
										 R(2,0),R(2,1),R(2,2),t(2));
			
			Pmats[std::make_pair(0,i)] = P1;
			
			std::vector<cv::KeyPoint> pt_set1,pt_set2;
			GetAlignedPointsFromMatch(imgpts[0],imgpts[i],matches,pt_set1,pt_set2);

			unsigned int start_i = pcloud.size();
			
			//adding more triangulated points to general cloud
			double reproj_error = TriangulatePoints(pt_set1, pt_set2, Kinv, P, P1, pcloud, correspImg1Pt);
			std::cout << "triangulation reproj error " << reproj_error << std::endl;
			std::cout << "before triangulation: " << start_i << " after " << pcloud.size() << std::endl;
			
			for (unsigned int j = 0; j<matches.size(); j++) {
				pcloud[start_i + j].idx_in_imgpts_for_img = std::vector<int>(imgs.size(),-1);
				//matches[i] corresponds to pointcloud[i]
				pcloud[start_i + j].idx_in_imgpts_for_img[0] = matches[j].queryIdx;
				pcloud[start_i + j].idx_in_imgpts_for_img[i] = matches[j].trainIdx;
			}
			
		}
		
		for (unsigned int i=0; i<pcloud.size(); i++) {
			pointcloud.push_back(pcloud[i]);
			pointCloudRGB.push_back(imgs_orig[0].at<cv::Vec3b>(imgpts[0][pcloud[i].idx_in_imgpts_for_img[0]].pt));
		}
		
	}
};