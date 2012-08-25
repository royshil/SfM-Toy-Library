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

void DecomposeEssentialUsingHorn90(double _E[9], double _R1[9], double _R2[9], double _t1[3], double _t2[3]) {
	//from : http://people.csail.mit.edu/bkph/articles/Essential.pdf
	using namespace Eigen;

	Matrix3d E = Map<Matrix<double,3,3,RowMajor> >(_E);
	Matrix3d EEt = E * E.transpose();
	Vector3d e0e1 = E.col(0).cross(E.col(1)),e1e2 = E.col(1).cross(E.col(2)),e2e0 = E.col(2).cross(E.col(2));
	Vector3d b1,b2;

#if 1
	//Method 1
	Matrix3d bbt = 0.5 * EEt.trace() * Matrix3d::Identity() - EEt; //Horn90 (12)
	Vector3d bbt_diag = bbt.diagonal();
	if (bbt_diag(0) > bbt_diag(1) && bbt_diag(0) > bbt_diag(2)) {
		b1 = bbt.row(0) / sqrt(bbt_diag(0));
		b2 = -b1;
	} else if (bbt_diag(1) > bbt_diag(0) && bbt_diag(1) > bbt_diag(2)) {
		b1 = bbt.row(1) / sqrt(bbt_diag(1));
		b2 = -b1;
	} else {
		b1 = bbt.row(2) / sqrt(bbt_diag(2));
		b2 = -b1;
	}
#else
	//Method 2
	if (e0e1.norm() > e1e2.norm() && e0e1.norm() > e2e0.norm()) {
		b1 = e0e1.normalized() * sqrt(0.5 * EEt.trace()); //Horn90 (18)
		b2 = -b1;
	} else if (e1e2.norm() > e0e1.norm() && e1e2.norm() > e2e0.norm()) {
		b1 = e1e2.normalized() * sqrt(0.5 * EEt.trace()); //Horn90 (18)
		b2 = -b1;
	} else {
		b1 = e2e0.normalized() * sqrt(0.5 * EEt.trace()); //Horn90 (18)
		b2 = -b1;
	}
#endif
	
	//Horn90 (19)
	Matrix3d cofactors; cofactors.col(0) = e1e2; cofactors.col(1) = e2e0; cofactors.col(2) = e0e1;
	cofactors.transposeInPlace();
	
	//B = [b]_x , see Horn90 (6) and http://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication
	Matrix3d B1; B1 <<	0,-b1(2),b1(1),
						b1(2),0,-b1(0),
						-b1(1),b1(0),0;
	Matrix3d B2; B2 <<	0,-b2(2),b2(1),
						b2(2),0,-b2(0),
						-b2(1),b2(0),0;

	Map<Matrix<double,3,3,RowMajor> > R1(_R1),R2(_R2);

	//Horn90 (24)
	R1 = (cofactors.transpose() - B1*E) / b1.dot(b1);
	R2 = (cofactors.transpose() - B2*E) / b2.dot(b2);
	Map<Vector3d> t1(_t1),t2(_t2); 
	t1 = b1; t2 = b2;
	
	cout << "Horn90 provided " << endl << R1 << endl << "and" << endl << R2 << endl;
}

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
	
	if(fabsf(determinant(R))-1.0 > 1e-07) {
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
	double percentage = ((double)count / (double)pcloud.size());
	cout << count << "/" << pcloud.size() << " = " << percentage*100.0 << "% are in front of camera" << endl;
	if(percentage < 0.75)
		return false; //less than 75% of the points are in front of the camera

	//check for coplanarity of points
	{
		cv::Mat_<double> cldm(pcloud.size(),3);
		for(unsigned int i=0;i<pcloud.size();i++) {
			cldm.row(i)(0) = pcloud[i].pt.x;
			cldm.row(i)(1) = pcloud[i].pt.y;
			cldm.row(i)(2) = pcloud[i].pt.z;
		}
		cv::Mat_<double> mean;
		cv::PCA pca(cldm,mean,CV_PCA_DATA_AS_ROW);

		int num_inliers = 0;
		cv::Vec3d nrm = pca.eigenvectors.row(2); nrm = nrm / norm(nrm);
		cv::Vec3d x0 = pca.mean;
		double p_to_plane_thresh = pca.eigenvalues.at<double>(2);

		for (int i=0; i<pcloud.size(); i++) {
			Vec3d w = Vec3d(pcloud[i].pt) - x0;
			double D = fabs(nrm.dot(w));
			if(D < p_to_plane_thresh) num_inliers++;
		}

		cout << num_inliers << "/" << pcloud.size() << " are coplanar" << endl;
		if((double)num_inliers / (double)(pcloud.size()) > 0.85)
			return false;
	}

	return true;
}

#define DECOMPOSE_SVD

bool DecomposeEtoRandT(
	Mat_<double>& E,
	Mat_<double>& R1,
	Mat_<double>& R2,
	Mat_<double>& t1,
	Mat_<double>& t2) 
{
#ifdef DECOMPOSE_SVD
	//Using HZ E decomposition
	Mat svd_u, svd_vt, svd_w;
	TakeSVDOfE(E,svd_u,svd_vt,svd_w);

	//check if first and second singular values are the same (as they should be)
	double singular_values_ratio = fabsf(svd_w.at<double>(0) / svd_w.at<double>(1));
	if(singular_values_ratio>1.0) singular_values_ratio = 1.0/singular_values_ratio; // flip ratio to keep it [0,1]
	if (singular_values_ratio < 0.7) {
		cout << "singular values are too far apart\n";
		return false;
	}

	Matx33d W(0,-1,0,	//HZ 9.13
		1,0,0,
		0,0,1);
	Matx33d Wt(0,1,0,
		-1,0,0,
		0,0,1);
	R1 = svd_u * Mat(W) * svd_vt; //HZ 9.19
	R2 = svd_u * Mat(Wt) * svd_vt; //HZ 9.19
	t1 = svd_u.col(2); //u3
	t2 = -svd_u.col(2); //u3
#else
	//Using Horn E decomposition
	DecomposeEssentialUsingHorn90(E[0],R1[0],R2[0],t1[0],t2[0]);
	return true;
#endif
}

bool FindCameraMatrices(const Mat& K, 
						const Mat& Kinv, 
						const Mat& distcoeff,
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

		//according to http://en.wikipedia.org/wiki/Essential_matrix#Properties_of_the_essential_matrix
		if(fabsf(determinant(E)) > 1e-07) {
			cout << "det(E) != 0 : " << determinant(E) << "\n";
			P1 = 0;
			return false;
		}
		
		Mat_<double> R1(3,3);
		Mat_<double> R2(3,3);
		Mat_<double> t1(1,3);
		Mat_<double> t2(1,3);

		//decompose E to P' , HZ (9.19)
		{			
			if (!DecomposeEtoRandT(E,R1,R2,t1,t2)) return false;

			if(determinant(R1)+1.0 < 1e-09) {
				//according to http://en.wikipedia.org/wiki/Essential_matrix#Showing_that_it_is_valid
				cout << "det(R) == -1 ["<<determinant(R1)<<"]: flip E's sign" << endl;
				E = -E;
				DecomposeEtoRandT(E,R1,R2,t1,t2);
			}
			if (!CheckCoherentRotation(R1)) {
				cout << "resulting rotation is not coherent\n";
				P1 = 0;
				return false;
			}
			
			P1 = Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t1(0),
						 R1(1,0),	R1(1,1),	R1(1,2),	t1(1),
						 R1(2,0),	R1(2,1),	R1(2,2),	t1(2));
			cout << "Testing P1 " << endl << Mat(P1) << endl;
			
			vector<CloudPoint> pcloud; vector<KeyPoint> corresp;
			double reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, Kinv, distcoeff, P, P1, pcloud, corresp);
			double reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, Kinv, distcoeff, P1, P, pcloud, corresp);
			
			//check if pointa are triangulated --in front-- of cameras for all 4 ambiguations
			if (!TestTriangulation(pcloud) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
				P1 = Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t2(0),
							 R1(1,0),	R1(1,1),	R1(1,2),	t2(1),
							 R1(2,0),	R1(2,1),	R1(2,2),	t2(2));
				cout << "Testing P1 "<< endl << Mat(P1) << endl;

				pcloud.clear(); corresp.clear();
				reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, Kinv, distcoeff, P, P1, pcloud, corresp);
				reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, Kinv, distcoeff, P1, P, pcloud, corresp);
				
				if (!TestTriangulation(pcloud) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
					if (!CheckCoherentRotation(R2)) {
						cout << "resulting rotation is not coherent\n";
						P1 = 0;
						return false;
					}
					
					P1 = Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	t1(0),
								 R2(1,0),	R2(1,1),	R2(1,2),	t1(1),
								 R2(2,0),	R2(2,1),	R2(2,2),	t1(2));
					cout << "Testing P1 "<< endl << Mat(P1) << endl;

					pcloud.clear(); corresp.clear();
					reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, Kinv, distcoeff, P, P1, pcloud, corresp);
					reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, Kinv, distcoeff, P1, P, pcloud, corresp);
					
					if (!TestTriangulation(pcloud) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
						P1 = Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	t2(0),
									 R2(1,0),	R2(1,1),	R2(1,2),	t2(1),
									 R2(2,0),	R2(2,1),	R2(2,2),	t2(2));
						cout << "Testing P1 "<< endl << Mat(P1) << endl;

						pcloud.clear(); corresp.clear();
						reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, Kinv, distcoeff, P, P1, pcloud, corresp);
						reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, Kinv, distcoeff, P1, P, pcloud, corresp);
						
						if (!TestTriangulation(pcloud) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
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
