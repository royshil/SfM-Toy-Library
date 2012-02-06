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
 
void FindCameraMatrices(const Mat& K, 
						const Mat& Kinv, 
						const vector<Point2d>& imgpts1,
						const vector<Point2d>& imgpts2,
						vector<Point2d>& imgpts1_good,
						vector<Point2d>& imgpts2_good,
						Matx34d& P,
						Matx34d& P1
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
		
		Mat F = findFundamentalMat(imgpts1, imgpts2, FM_RANSAC, 1, 0.99, status);
		cout << "keeping " << countNonZero(status) << " / " << status.size() << endl;	
		for (unsigned int i=0; i<status.size(); i++) {
			if (status[i]) 
			{
				imgpts1_good.push_back(imgpts1[i]);
				imgpts2_good.push_back(imgpts2[i]);
#ifdef __SFM__DEBUG__
				good_matches_.push_back(DMatch(imgpts1_good.size()-1,imgpts1_good.size()-1,1.0));
				keypoints_1.push_back(KeyPoint(imgpts1[i],1));
				keypoints_2.push_back(KeyPoint(imgpts2[i],1));
#endif
			}
		}	
		
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
		
		Mat E = K.t() * F * K; //according to HZ (9.12)
		//decompose E to P' , HZ (9.19)
		//if(false)
		{			
			RNG rng;
			unsigned int idx = rng(imgpts1_good.size());
			Point2f kp = imgpts1_good[idx]; 
			Point3d u(kp.x,kp.y,1.0);
			Mat_<double> um = Kinv * Mat_<double>(u); 
			u = um.at<Point3d>(0);
			Point2f kp1 = imgpts2_good[idx];
			Point3d u1(kp1.x,kp1.y,1.0);
			Mat_<double> um1 = Kinv * Mat_<double>(u1); 
			u1 = um1.at<Point3d>(0);
			
			SVD svd(E);
			//		Mat_<double> W = (Mat_<double>(3,3) << svd.w.at<double>(0),0,0, 0,svd.w.at<double>(1),0, 0,0,svd.w.at<double>(2));
			Matx33d W(0,-1,0,	//HZ 9.13
					  1,0,0,
					  0,0,1);
			Matx33d Winv(0,1,0,
						 -1,0,0,
						 0,0,1);
			Mat_<double> R = svd.u * Mat(W) * svd.vt; //HZ 9.19
			Mat_<double> t = svd.u.col(2); //u3
			//		Mat_<double> t = (Mat_<double>(1,3) << 30,0,0);
			cout << "Testing P1 " << endl << R << endl << t << endl;
			P1 = Matx34d(R(0,0),	R(0,1),	R(0,2),	t(0),
						 R(1,0),	R(1,1),	R(1,2),	t(1),
						 R(2,0),	R(2,1),	R(2,2), t(2));
			
			Mat_<double> X = IterativeLinearLSTriangulation(u,P,u1,P1);
			
			//check if point is in front of cameras for all 4 configurations
			if (X(2) < 0) {
				t = -t;
				P1 = Matx34d(R(0,0),	R(0,1),	R(0,2),	t(0),
							 R(1,0),	R(1,1),	R(1,2),	t(1),
							 R(2,0),	R(2,1),	R(2,2), t(2));
				cout << "Testing P1 " << endl << R << endl << t << endl;
				X = IterativeLinearLSTriangulation(u,P,u1,P1);
				
				if (X(2) < 0) {
					t = -t;
					R = svd.u * Mat(Winv) * svd.vt;
					P1 = Matx34d(R(0,0),	R(0,1),	R(0,2),	t(0),
								 R(1,0),	R(1,1),	R(1,2),	t(1),
								 R(2,0),	R(2,1),	R(2,2), t(2));
					cout << "Testing P1 " << endl << R << endl << t << endl;
					X = IterativeLinearLSTriangulation(u,P,u1,P1);
					
					if (X(2) < 0) {
						t = -t;
						P1 = Matx34d(R(0,0),	R(0,1),	R(0,2),	t(0),
									 R(1,0),	R(1,1),	R(1,2),	t(1),
									 R(2,0),	R(2,1),	R(2,2), t(2));
						cout << "Testing P1 " << endl << R << endl << t << endl;
						X = IterativeLinearLSTriangulation(u,P,u1,P1);
						if (X(2) < 0) {
							cout << "Shit." << endl; exit(0);
						}
					}				
				}			
			}
		}
		
		t = ((double)getTickCount() - t)/getTickFrequency();
		cout << "Done. (" << t <<"s)"<< endl;
	}
}