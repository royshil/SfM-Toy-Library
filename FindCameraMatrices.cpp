/*
 *  FindCameraMatrices.cpp
 *  EyeRingOpenCV
 *
 *  Created by Roy Shilkrot on 12/23/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */

#include "FindCameraMatrices.h"
#include "Triangulation.h"

#include <vector>
#include <iostream>

#include <opencv2/calib3d/calib3d.hpp>

using namespace cv;
using namespace std;

//#define USE_EIGEN 1

#ifdef USE_EIGEN
#include <eigen3/Eigen/Eigen>

#endif

bool CheckCoherentRotation(cv::Mat_<double>& R) {
	std::cout << "R; " << R << std::endl;
	double s = cv::norm(cv::abs(R),cv::Mat_<double>::eye(3,3),cv::NORM_L1);
	std::cout << "Distance from I: " << s << std::endl;
	if (s > 2.3) { // norm of R from I is large -> probably bad rotation
		std::cout << "rotation is probably not coherent.." << std::endl;
		return false;	//skip triangulation
	}
	return true;
}

void FindCameraMatrices(const Mat& K, 
						const Mat& Kinv, 
						const vector<KeyPoint>& imgpts1,
						const vector<KeyPoint>& imgpts2,
						vector<KeyPoint>& imgpts1_good,
						vector<KeyPoint>& imgpts2_good,
						Matx34d& P,
						Matx34d& P1,
						vector<DMatch>& matches,
						vector<CloudPoint>& outCloud
#ifdef __SFM__DEBUG__
						,const Mat& img_1,
						const Mat& img_2
#endif
						) 
{
	//Find camera matrices
	{
		cout << "Find camera matrices...";
		double t = getTickCount();
		//Try to eliminate keypoints based on the fundamental matrix
		//(although this is not the proper way to do this)
		vector<uchar> status(imgpts1.size());
		
#ifdef __SFM__DEBUG__
		std::vector< DMatch > good_matches_;
		std::vector<KeyPoint> keypoints_1, keypoints_2;
#endif		
		//	undistortPoints(imgpts1, imgpts1, cam_matrix, distortion_coeff);
		//	undistortPoints(imgpts2, imgpts2, cam_matrix, distortion_coeff);
		//
		imgpts1_good.clear(); imgpts2_good.clear();
		
		vector<KeyPoint> imgpts1_tmp;
		vector<KeyPoint> imgpts2_tmp;
		if (matches.size() <= 0) {
			imgpts1_tmp = imgpts1;
			imgpts2_tmp = imgpts2;
		} else {
			GetAlignedPointsFromMatch(imgpts1, imgpts2, matches, imgpts1_tmp, imgpts2_tmp);
//			for (unsigned int i=0; i<matches.size(); i++) {
//				imgpts1_tmp.push_back(imgpts1[matches[i].queryIdx]);
//				imgpts2_tmp.push_back(imgpts2[matches[i].trainIdx]);
//			}
		}
		
		Mat F;
		{
			vector<Point2f> pts1,pts2;
			KeyPointsToPoints(imgpts1_tmp, pts1);
			KeyPointsToPoints(imgpts2_tmp, pts2);
#ifdef __SFM__DEBUG__
			cout << "pts1 " << pts1.size() << " (orig pts " << imgpts1_good.size() << ")" << endl;
			cout << "pts2 " << pts2.size() << " (orig pts " << imgpts2_good.size() << ")" << endl;
#endif
			F = findFundamentalMat(pts1, pts2, FM_RANSAC, 3.0, 0.99, status);
		}

		vector<DMatch> new_matches;
		cout << "keeping " << countNonZero(status) << " / " << status.size() << endl;	
		for (unsigned int i=0; i<status.size(); i++) {
			if (status[i]) 
			{
				imgpts1_good.push_back(imgpts1_tmp[i]);
				imgpts2_good.push_back(imgpts2_tmp[i]);
				new_matches.push_back(DMatch(matches[i].queryIdx,matches[i].trainIdx,1.0));
#ifdef __SFM__DEBUG__
				good_matches_.push_back(DMatch(imgpts1_good.size()-1,imgpts1_good.size()-1,1.0));
				keypoints_1.push_back(imgpts1_tmp[i]);
				keypoints_2.push_back(imgpts2_tmp[i]);
#endif
			}
		}	
		
		cout << matches.size() << " matches before, " << new_matches.size() << " new matches\n";
		matches = new_matches; //keep only those points who survived the fundamental matrix
		
		//-- Draw only "good" matches
#ifdef __SFM__DEBUG__
		{
			Mat img_matches;
			drawMatches( img_1, keypoints_1, img_2, keypoints_2,
						good_matches_, img_matches, Scalar::all(-1), Scalar::all(-1),
						vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );		
			//-- Show detected matches
			imshow( "Good Matches", img_matches );
			waitKey(0);
			destroyWindow("Good Matches");
		}
#endif		
		
		
		//Essential matrix: compute then extract cameras [R|t]
		
		Mat_<double> E = K.t() * F * K; //according to HZ (9.12)
		//decompose E to P' , HZ (9.19)
		//if(false)
		{			
//			RNG rng;
//			unsigned int idx = rng(imgpts1_good.size());
//			Point2f kp = imgpts1_good[idx].pt; 
//			Point3d u(kp.x,kp.y,1.0);
//			Mat_<double> um = Kinv * Mat_<double>(u); 
//			u = um.at<Point3d>(0);
//			Point2f kp1 = imgpts2_good[idx].pt;
//			Point3d u1(kp1.x,kp1.y,1.0);
//			Mat_<double> um1 = Kinv * Mat_<double>(u1); 
//			u1 = um1.at<Point3d>(0);
			
#ifndef USE_EIGEN
			SVD svd(E,SVD::MODIFY_A);
			Mat svd_u = svd.u;
			Mat svd_vt = svd.vt;
			Mat svd_w = svd.w;
#else
			cout << "Eigen3 SVD..\n";
			Eigen::Matrix3f  e;   e << E(0,0), E(0,1), E(0,2),
										E(1,0), E(1,1), E(1,2),
										E(2,0), E(2,1), E(2,2);
			Eigen::JacobiSVD<Eigen::MatrixXf> svd(e, Eigen::ComputeThinU | Eigen::ComputeThinV);
			Eigen::MatrixXf Esvd_u = svd.matrixU();
			Eigen::MatrixXf Esvd_v = svd.matrixV();
			Mat_<double> svd_u = (Mat_<double>(3,3) << Esvd_u(0,0), Esvd_u(0,1), Esvd_u(0,2),
														Esvd_u(1,0), Esvd_u(1,1), Esvd_u(1,2), 
														Esvd_u(2,0), Esvd_u(2,1), Esvd_u(2,2)); 
			Mat_<double> svd_v = (Mat_<double>(3,3) << Esvd_v(0,0), Esvd_v(0,1), Esvd_v(0,2),
														Esvd_v(1,0), Esvd_v(1,1), Esvd_v(1,2), 
														Esvd_v(2,0), Esvd_v(2,1), Esvd_v(2,2));
			Mat svd_vt = svd_v.t();
			Mat_<double> svd_w = (Mat_<double>(1,3) << svd.singularValues()[0] , svd.singularValues()[1] , svd.singularValues()[2]);
#endif
			
			cout << "----------------------- SVD ------------------------\n";
			cout << "U:\n"<<svd_u<<"\nW:\n"<<svd_w<<"\nVt:\n"<<svd_vt<<endl;
			cout << "----------------------------------------------------\n";

//			if (fabsf(svd_w.at<double>(0) - svd_w.at<double>(1)) > 0.75) {
//				cout << "singular values are too far apart\n";
//				P1 = 0; 
//				return;
//			}
			
			Matx33d W(0,-1,0,	//HZ 9.13
					  1,0,0,
					  0,0,1);
			Matx33d Wt(0,1,0,
					   -1,0,0,
					   0,0,1);
			Mat_<double> R = svd_u * Mat(W) * svd_vt; //HZ 9.19
			Mat_<double> t = svd_u.col(2); //u3
			
			if (!CheckCoherentRotation(R)) {
				cout << "resulting rotation is not coherent\n";
				P1 = 0;
				return;
			}			
			
			P1 = Matx34d(R(0,0),	R(0,1),	R(0,2),	t(0),
						 R(1,0),	R(1,1),	R(1,2),	t(1),
						 R(2,0),	R(2,1),	R(2,2), t(2));
			cout << "Testing P1 " << endl << Mat(P1) << endl;
			
			vector<CloudPoint> pcloud; vector<KeyPoint> corresp;
			TriangulatePoints(imgpts1_good, imgpts2_good, Kinv, P, P1, pcloud, corresp);
			Scalar X = mean(CloudPointsToPoints(pcloud));
			cout <<	"Mean :" << X[0] << "," << X[1] << "," << X[2] << "," << X[3]  << endl;
			
			//check if point is in front of cameras for all 4 configurations
			if (X(2) < 0) {
				t = -svd_u.col(2); //-u3
				P1 = Matx34d(R(0,0),	R(0,1),	R(0,2),	t(0),
							 R(1,0),	R(1,1),	R(1,2),	t(1),
							 R(2,0),	R(2,1),	R(2,2), t(2));
				cout << "Testing P1 "<< endl << Mat(P1) << endl;

				pcloud.clear(); corresp.clear();
				TriangulatePoints(imgpts1_good, imgpts2_good, Kinv, P, P1, pcloud, corresp);
				X = mean(pcloud);
				cout <<	"Mean :" << X[0] << "," << X[1] << "," << X[2] << "," << X[3]  << endl;
				
				if (X(2) < 0) {
					t = svd_u.col(2); //u3
					R = svd_u * Mat(Wt) * svd_vt;
					P1 = Matx34d(R(0,0),	R(0,1),	R(0,2),	t(0),
								 R(1,0),	R(1,1),	R(1,2),	t(1),
								 R(2,0),	R(2,1),	R(2,2), t(2));
					cout << "Testing P1 "<< endl << Mat(P1) << endl;

					pcloud.clear(); corresp.clear();
					TriangulatePoints(imgpts1_good, imgpts2_good, Kinv, P, P1, pcloud, corresp);
					X = mean(pcloud);
					cout <<	"Mean :" << X[0] << "," << X[1] << "," << X[2] << "," << X[3]  << endl;
					
					if (X(2) < 0) {
						t = -svd_u.col(2);//-u3
						P1 = Matx34d(R(0,0),	R(0,1),	R(0,2),	t(0),
									 R(1,0),	R(1,1),	R(1,2),	t(1),
									 R(2,0),	R(2,1),	R(2,2), t(2));
						cout << "Testing P1 "<< endl << Mat(P1) << endl;

						pcloud.clear(); corresp.clear();
						TriangulatePoints(imgpts1_good, imgpts2_good, Kinv, P, P1, pcloud, corresp);
						X = mean(pcloud);
						cout <<	"Mean :" << X[0] << "," << X[1] << "," << X[2] << "," << X[3]  << endl;
						
						if (X(2) < 0) {
							cout << "Shit." << endl; exit(0);
						}
					}				
				}			
			}
			for (unsigned int i=0; i<pcloud.size(); i++) {
				outCloud.push_back(pcloud[i]);
			}
		}		
		
		t = ((double)getTickCount() - t)/getTickFrequency();
		cout << "Done. (" << t <<"s)"<< endl;
	}
}