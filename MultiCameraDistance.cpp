/*
 *  MultiCameraDistance.cpp
 *  SfMToyExample
 *
 *  Created by Roy Shilkrot on 3/27/12.
 *  Copyright 2012 MIT. All rights reserved.
 *
 */

#include "MultiCameraDistance.h"
	//c'tor
MultiCameraDistance::MultiCameraDistance(const std::vector<cv::Mat>& imgs_, const std::vector<std::string>& imgs_names_):
	features_matched(false),
	imgs_names(imgs_names_)
	{		
		for (unsigned int i=0; i<imgs_.size(); i++) {
			imgs_orig.push_back(cv::Mat());
			if (!imgs_[i].empty()) {
				if (imgs_[i].type() == CV_8UC1) {
					cvtColor(imgs_[i], imgs_orig[i], CV_GRAY2BGR);
				} else if (imgs_[i].type() == CV_32FC3 || imgs_[i].type() == CV_64FC3) {
					imgs_[i].convertTo(imgs_orig[i],CV_8UC1,255.0);
				} else {
					imgs_[i].copyTo(imgs_orig[i]);
				}
				
			}
			
			imgs.push_back(cv::Mat());
			cvtColor(imgs_orig[i],imgs[i], CV_BGR2GRAY);
			
			//			imgpts.push_back(std::vector<cv::KeyPoint>());
			//			fullpts.push_back(std::vector<cv::KeyPoint>());
			imgpts_good.push_back(std::vector<cv::KeyPoint>());
			//			descriptors.push_back(cv::Mat());
		}
		
		cv::SurfFeatureDetector detector( 10 );
		detector.detect(imgs, imgpts);
		cv::SurfDescriptorExtractor extractor(48,16,true);
		extractor.compute(imgs, imgpts, descriptors);
		
		//		cv::FileStorage fs;
		//		fs.open("../camera.yaml",cv::FileStorage::READ);
		//		fs["camera_matrix"]>>cam_matrix;
		//		fs["distortion_coefficients"]>>distortion_coeff;
		
		cv::Size imgs_size = imgs_[0].size();
		cam_matrix = (cv::Mat_<double>(3,3) << imgs_size.height , 0 , imgs_size.width/2.0,
					  0, imgs_size.height, imgs_size.height/2.0,
					  0, 0, 1);
		
		K = cam_matrix;
		invert(K, Kinv); //get inverse of camera matrix
	}
	
	void MultiCameraDistance::OnlyMatchFeatures(int strategy) 
	{
		if(features_matched) return;
		int loop1_top = imgs.size() - 1, loop2_top = imgs.size();
		int frame_num_i = 0;
		//#pragma omp parallel for schedule(dynamic)
		for (frame_num_i = 0; frame_num_i < loop1_top; frame_num_i++) {
			for (int frame_num_j = frame_num_i + 1; frame_num_j < loop2_top; frame_num_j++)
			{
				std::vector<cv::KeyPoint> fp,fp1;
				std::cout << "------------ Match " << imgs_names[frame_num_i] << ","<<imgs_names[frame_num_j]<<" ------------\n";
				std::vector<cv::DMatch> matches_tmp;
				MatchFeatures(imgs[frame_num_i], imgs_orig[frame_num_i], 
							  imgs[frame_num_j], imgs_orig[frame_num_j],
							  imgpts[frame_num_i],
							  imgpts[frame_num_j],
							  descriptors[frame_num_i],
							  descriptors[frame_num_j],
							  fp,
							  fp1,
							  STRATEGY_USE_FEATURE_MATCH,
							  &matches_tmp);
				
				//#pragma omp critical
				{
					matches_matrix[std::make_pair(frame_num_i,frame_num_j)] = matches_tmp;
				}
			}
		}
		features_matched = true;
	}
	
	bool MultiCameraDistance::CheckCoherentRotation(cv::Mat_<double>& R) {
		std::cout << "R; " << R << std::endl;
		double s = cv::norm(R,cv::Mat_<double>::eye(3,3),cv::NORM_L1);
		std::cout << "Distance from I: " << s << std::endl;
		if (s > 2.3) { // norm of R from I is large -> probably bad rotation
			std::cout << "rotation is probably not coherent.." << std::endl;
			return false;	//skip triangulation
		}
		return true;
	}
	
	void MultiCameraDistance::RecoverDepthFromImages() {			
		
		if(!features_matched) 
			OnlyMatchFeatures();
		
		std::cout << "======================================================================\n";
		std::cout << "======================================================================\n";
		
		std::vector<CloudPoint> pcloud_;
		for (unsigned int frame_num_i = 0; frame_num_i < imgs.size() - 1; frame_num_i++) {
			for (unsigned int frame_num_j = frame_num_i + 1; frame_num_j < imgs.size(); frame_num_j++) 
			{				
				cv::Matx34d P(1,0,0,0,
							  0,1,0,0,
							  0,0,1,0);
				
				std::cout << "---------------- find Ps: "<<imgs_names[frame_num_i]<<","<<imgs_names[frame_num_j]<<" -----------------\n";
				//TODO: obtain camera matrices from pairwise matches
				FindCameraMatrices(K, Kinv, 
								   imgpts[frame_num_i], 
								   imgpts[frame_num_j], 
								   imgpts_good[frame_num_i],
								   imgpts_good[frame_num_j], 
								   P, 
								   Pmats[std::make_pair(frame_num_i, frame_num_j)],
								   matches_matrix[std::make_pair(frame_num_i, frame_num_j)],
								   pcloud_
#ifdef __SFM__DEBUG__
								   ,imgs[frame_num_i],imgs[frame_num_j]
#endif
								   );
			}
		}
		
		std::cout << "======================================================================\n";
		std::cout << "======================================================================\n";
		
		for (
			 unsigned int frame_num_i = 0; 
			 frame_num_i < imgs.size() - 1; 
			 frame_num_i++) 
		{
			for (
				 unsigned int frame_num_j = frame_num_i + 1; 
				 frame_num_j < imgs.size(); 
				 frame_num_j++) 
			{
				std::cout << "------------ triangulate "<<imgs_names[frame_num_i]<<","<<imgs_names[frame_num_j]<<"-------------\n";
				
				//TODO: if the P1 matrix is far away from identity rotation - the solution is probably invalid...
				//so use an identity matrix
				cv::Matx34d P1 = Pmats[std::make_pair(frame_num_i, frame_num_j)];
				cv::Mat_<double> R = (cv::Mat_<double>(3,3) << P1(0,0),P1(0,1),P1(0,2),
									  P1(1,0),P1(1,1),P1(1,2),
									  P1(2,0),P1(2,1),P1(2,2));
				if(!CheckCoherentRotation(R)) {
					std::cout << " skip triangulation " << std::endl;
					continue; //skip triangulation 
				}
				
				std::vector<cv::KeyPoint> pt_set1,pt_set2;
//				std::vector<cv::DMatch> matches = matches_matrix[std::make_pair(frame_num_i, frame_num_j)];
//				for (unsigned int i=0; i<matches.size(); i++) {
//					pt_set1.push_back(imgpts[frame_num_i][matches[i].queryIdx]);
//					pt_set2.push_back(imgpts[frame_num_j][matches[i].trainIdx]);
//				}
				GetAlignedPointsFromMatch(imgpts[frame_num_i],
										  imgpts[frame_num_j],
										  matches_matrix[std::make_pair(frame_num_i, frame_num_j)],
										  pt_set1,
										  pt_set2);
				
				//-- triangulate points for each pair, and transform to base frame
				
#if 0								
				//if this is not between some frame and the 1st camera...
				if (frame_num_i > 0) 
				{
					std::cout << "this is not reference frame, find backtrack\n";
					//get rotation to first camera					
					cv::Matx34d P1_ref;
					cv::Mat_<double> R_ref;
					int reference_frame = 0; //start by trying to reference to origin frame
					do {
						std::cout << "Check frame " << reference_frame << "\n";
						P1_ref = Pmats[std::make_pair(reference_frame++, frame_num_i)];
						R_ref = (cv::Mat_<double>(3,3) << P1_ref(0,0),P1_ref(0,1),P1_ref(0,2),
								 P1_ref(1,0),P1_ref(1,1),P1_ref(1,2),
								 P1_ref(2,0),P1_ref(2,1),P1_ref(2,2));
					} while (!CheckCoherentRotation(R_ref) && reference_frame < frame_num_i); //see if rotation from the i cam to the reference is bad.
					reference_frame--; //undo the ++ from before
					
					std::cout << "Ps: " << reference_frame << " -> " << frame_num_i << " -> " << frame_num_j << std::endl;
					
					R = R_ref * R;
					cv::Matx31d t(P1(0,3),P1(1,3),P1(2,3));
					cv::Matx31d t_ref(P1_ref(0,3),P1_ref(1,3),P1_ref(2,3));
					t = t + t_ref;
					P1 = cv::Matx34d(R(0,0),R(0,1),R(0,2),t(0),
									 R(1,0),R(1,1),R(1,2),t(1),
									 R(2,0),R(2,1),R(2,2),t(2));
				} 
#endif
				R = (cv::Mat_<double>(3,3) << P1(0,0),P1(0,1),P1(0,2),
					 P1(1,0),P1(1,1),P1(1,2),
					 P1(2,0),P1(2,1),P1(2,2));
				if(!CheckCoherentRotation(R)) {
					std::cout << " skip triangulation " << std::endl;
					continue; //skip triangulation 
				}
				
				//triangulate
				std::vector<CloudPoint> pcloud;
				correspImg1Pt.clear();
				cv::Matx34d P(1,0,0,0,
							  0,1,0,0,
							  0,0,1,0);				
				std::cout << "P1 " << cv::Mat(P1) << std::endl;
				double reproj_error = TriangulatePoints(pt_set1, pt_set2, Kinv, P, P1, pcloud, correspImg1Pt);
				std::cout << "triangulation reproj error " << reproj_error << std::endl;
				
				if (reproj_error < 500.0) {
					for (unsigned int i=0; i<pcloud.size(); i++) {
						pointcloud.push_back(pcloud[i]);
						pointCloudRGB.push_back(imgs_orig[frame_num_i].at<cv::Vec3b>(correspImg1Pt[i].pt));
					}
				} else {
					std::cout << "triangulation mean reproj. error is too high" << std::endl;
				}
			}
		}
	}