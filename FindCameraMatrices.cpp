/*
 *  FindCameraMatrices.cpp
 *  SfM-Toy-Library
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

#include <Eigen/Eigen>

bool CheckCoherentRotation(cv::Mat_<double>& R) {
	std::cout << "R; " << R << std::endl;
	//double s = cv::norm(cv::abs(R),cv::Mat_<double>::eye(3,3),cv::NORM_L1);
	//std::cout << "Distance from I: " << s << std::endl;
	//if (s > 2.3) { // norm of R from I is large -> probably bad rotation
	//	std::cout << "rotation is probably not coherent.." << std::endl;
	//	return false;	//skip triangulation
	//}
	//Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor> > eR(R[0]);
	//if(eR(2,0) < -0.9)
	//{
	//	cout << "rotate 180deg (PI rad) on Y" << endl;

	//	cout << "before" << endl << eR << endl;
	//	Eigen::AngleAxisd aad(-M_PI/2.0,Eigen::Vector3d::UnitY());
	//	eR *= aad.toRotationMatrix();
	//	cout << "after" << endl << eR << endl;
	//}
	//if(eR(0,0) < -0.9) {
	//	cout << "flip right vector" << endl;
	//	eR.row(0) = -eR.row(0);
	//}
	
	if(fabsf(determinant(R))-1.0 > 1e-09) {
		cerr << "det(R) != +-1.0, this is not a rotation matrix" << endl;
		return false;
	}

	return true;
}

Mat GetFundamentalMat(const vector<KeyPoint>& imgpts1,
					   const vector<KeyPoint>& imgpts2,
					   vector<KeyPoint>& imgpts1_good,
					   vector<KeyPoint>& imgpts2_good,
					   vector<DMatch>& matches
#ifdef __SFM__DEBUG__
					  ,const Mat& img_1,
					  const Mat& img_2
#endif
					  ) 
{
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
	cout << "F keeping " << countNonZero(status) << " / " << status.size() << endl;	
	for (unsigned int i=0; i<status.size(); i++) {
		if (status[i]) 
		{
			imgpts1_good.push_back(imgpts1_tmp[i]);
			imgpts2_good.push_back(imgpts2_tmp[i]);

			new_matches.push_back(DMatch(matches[i].queryIdx,matches[i].trainIdx,matches[i].distance));
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
	
	return F;
}

void TakeSVDOfE(Mat& E, Mat& svd_u, Mat& svd_vt, Mat& svd_w) {
#ifndef USE_EIGEN
	SVD svd(E,SVD::MODIFY_A);
	svd_u = svd.u;
	svd_vt = svd.vt;
	svd_w = svd.w;
#else
	cout << "Eigen3 SVD..\n";
	Eigen::Matrix3f  e;   e << E(0,0), E(0,1), E(0,2),
	E(1,0), E(1,1), E(1,2),
	E(2,0), E(2,1), E(2,2);
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(e, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::MatrixXf Esvd_u = svd.matrixU();
	Eigen::MatrixXf Esvd_v = svd.matrixV();
	svd_u = (Mat_<double>(3,3) << Esvd_u(0,0), Esvd_u(0,1), Esvd_u(0,2),
						  Esvd_u(1,0), Esvd_u(1,1), Esvd_u(1,2), 
						  Esvd_u(2,0), Esvd_u(2,1), Esvd_u(2,2)); 
	Mat_<double> svd_v = (Mat_<double>(3,3) << Esvd_v(0,0), Esvd_v(0,1), Esvd_v(0,2),
						  Esvd_v(1,0), Esvd_v(1,1), Esvd_v(1,2), 
						  Esvd_v(2,0), Esvd_v(2,1), Esvd_v(2,2));
	svd_vt = svd_v.t();
	svd_w = (Mat_<double>(1,3) << svd.singularValues()[0] , svd.singularValues()[1] , svd.singularValues()[2]);
#endif
	
	cout << "----------------------- SVD ------------------------\n";
	cout << "U:\n"<<svd_u<<"\nW:\n"<<svd_w<<"\nVt:\n"<<svd_vt<<endl;
	cout << "----------------------------------------------------\n";
}

bool TestTriangulation(const vector<CloudPoint>& pcloud) {
	int count = 0;
	for (int i=0; i<pcloud.size(); i++) {
		count += pcloud[i].pt.z > 0 ? 1 : 0;
	}
	cout << count << "/" << pcloud.size() << " = " << (count / pcloud.size())*100 << "% are in front of camera" << endl;
	return (count / pcloud.size()) < 0.8; //allow only 20% "outliers"
}

bool FindCameraMatrices(const Mat& K, 
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
		
		Mat F = GetFundamentalMat(imgpts1,imgpts2,imgpts1_good,imgpts2_good,matches
#ifdef __SFM__DEBUG__
								  ,img_1,img_2
#endif
								  );
		
		//Essential matrix: compute then extract cameras [R|t]
		Mat_<double> E = K.t() * F * K; //according to HZ (9.12)
		//decompose E to P' , HZ (9.19)
		{			
			Mat svd_u, svd_vt, svd_w;
			TakeSVDOfE(E,svd_u,svd_vt,svd_w);

			//check if first and second singular values are the same (as they should be)
			double singular_values_ratio = fabsf(svd_w.at<double>(0) / svd_w.at<double>(1));
			if(singular_values_ratio>1.0) singular_values_ratio = 1.0/singular_values_ratio; // flip ratio to keep it [0,1]
			if (singular_values_ratio < 0.7) {
				cout << "singular values are too far apart\n";
				P1 = 0; 
				return false;
			}
			
			//according to http://en.wikipedia.org/wiki/Essential_matrix#Properties_of_the_essential_matrix
			if(fabsf(determinant(E)) > 1e-08) {
				cout << "det(E) != 0 : " << determinant(E) << "\n";
				P1 = 0;
				return false;
			}
									
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
				return false;
			}
			
			if(determinant(R)+1.0 < 1e-09) {
				//according to http://en.wikipedia.org/wiki/Essential_matrix#Showing_that_it_is_valid
				cout << "det(R) == -1 ["<<determinant(R)<<"]: flip E's sign" << endl;
				E = -E;
				TakeSVDOfE(E, svd_u, svd_vt, svd_w);
				R = svd_u * Mat(W) * svd_vt;
				t = svd_u.col(2);
			}
			
			P1 = Matx34d(R(0,0),	R(0,1),	R(0,2),	t(0),
						 R(1,0),	R(1,1),	R(1,2),	t(1),
						 R(2,0),	R(2,1),	R(2,2), t(2));
			cout << "Testing P1 " << endl << Mat(P1) << endl;
			
			vector<CloudPoint> pcloud; vector<KeyPoint> corresp;
			TriangulatePoints(imgpts1_good, imgpts2_good, Kinv, P, P1, pcloud, corresp);
//			Scalar X = mean(CloudPointsToPoints(pcloud));
//			cout <<	"Mean :" << X[0] << "," << X[1] << "," << X[2] << "," << X[3]  << endl;
			
			//check if pointa are triangulated --in front-- of cameras for all 4 ambiguations
			if (!TestTriangulation(pcloud)) {
				t = -svd_u.col(2); //-u3
				P1 = Matx34d(R(0,0),	R(0,1),	R(0,2),	t(0),
							 R(1,0),	R(1,1),	R(1,2),	t(1),
							 R(2,0),	R(2,1),	R(2,2), t(2));
				cout << "Testing P1 "<< endl << Mat(P1) << endl;

				pcloud.clear(); corresp.clear();
				TriangulatePoints(imgpts1_good, imgpts2_good, Kinv, P, P1, pcloud, corresp);
//				X = mean(CloudPointsToPoints(pcloud));
//				cout <<	"Mean :" << X[0] << "," << X[1] << "," << X[2] << "," << X[3]  << endl;
				
				if (!TestTriangulation(pcloud)) {
					t = svd_u.col(2); //u3
					R = svd_u * Mat(Wt) * svd_vt; //UWtVt
					P1 = Matx34d(R(0,0),	R(0,1),	R(0,2),	t(0),
								 R(1,0),	R(1,1),	R(1,2),	t(1),
								 R(2,0),	R(2,1),	R(2,2), t(2));
					cout << "Testing P1 "<< endl << Mat(P1) << endl;

					pcloud.clear(); corresp.clear();
					TriangulatePoints(imgpts1_good, imgpts2_good, Kinv, P, P1, pcloud, corresp);
//					X = mean(CloudPointsToPoints(pcloud));
//					cout <<	"Mean :" << X[0] << "," << X[1] << "," << X[2] << "," << X[3]  << endl;
					
					if (!TestTriangulation(pcloud)) {
						t = -svd_u.col(2);//-u3
						P1 = Matx34d(R(0,0),	R(0,1),	R(0,2),	t(0),
									 R(1,0),	R(1,1),	R(1,2),	t(1),
									 R(2,0),	R(2,1),	R(2,2), t(2));
						cout << "Testing P1 "<< endl << Mat(P1) << endl;

						pcloud.clear(); corresp.clear();
						TriangulatePoints(imgpts1_good, imgpts2_good, Kinv, P, P1, pcloud, corresp);
//						X = mean(CloudPointsToPoints(pcloud));
//						cout <<	"Mean :" << X[0] << "," << X[1] << "," << X[2] << "," << X[3]  << endl;
						
						if (!TestTriangulation(pcloud)) {
							cout << "Shit." << endl; 
							return false;
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
	return true;
}
