/*
 *  Triangulation.cpp
 *  EyeRingOpenCV
 *
 *  Created by Roy Shilkrot on 12/23/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */

#include "Triangulation.h"

#include <iostream>

using namespace std;
using namespace cv;


/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
Mat_<double> LinearLSTriangulation(Point3d u,		//homogenous image point (u,v,1)
								   Matx34d P,		//camera 1 matrix
								   Point3d u1,		//homogenous image point in 2nd camera
								   Matx34d P1		//camera 2 matrix
								   ) 
{
	
	//build matrix A for homogenous equation system Ax = 0
	//assume X = (x,y,z,1), for Linear-LS method
	//which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
	//	cout << "u " << u <<", u1 " << u1 << endl;
	//	Matx<double,6,4> A; //this is for the AX=0 case, and with linear dependence..
	//	A(0) = u.x*P(2)-P(0);
	//	A(1) = u.y*P(2)-P(1);
	//	A(2) = u.x*P(1)-u.y*P(0);
	//	A(3) = u1.x*P1(2)-P1(0);
	//	A(4) = u1.y*P1(2)-P1(1);
	//	A(5) = u1.x*P(1)-u1.y*P1(0);
	//	Matx43d A; //not working for some reason...
	//	A(0) = u.x*P(2)-P(0);
	//	A(1) = u.y*P(2)-P(1);
	//	A(2) = u1.x*P1(2)-P1(0);
	//	A(3) = u1.y*P1(2)-P1(1);
	Matx43d A(u.x*P(2,0)-P(0,0),	u.x*P(2,1)-P(0,1),		u.x*P(2,2)-P(0,2),		
			  u.y*P(2,0)-P(1,0),	u.y*P(2,1)-P(1,1),		u.y*P(2,2)-P(1,2),		
			  u1.x*P1(2,0)-P1(0,0), u1.x*P1(2,1)-P1(0,1),	u1.x*P1(2,2)-P1(0,2),	
			  u1.y*P1(2,0)-P1(1,0), u1.y*P1(2,1)-P1(1,1),	u1.y*P1(2,2)-P1(1,2)
			  );
	Mat_<double> B = (Mat_<double>(4,1) <<	-(u.x*P(2,3)	-P(0,3)),
					  -(u.y*P(2,3)	-P(1,3)),
					  -(u1.x*P1(2,3)	-P1(0,3)),
					  -(u1.y*P1(2,3)	-P1(1,3)));
	
	Mat_<double> X;
	solve(A,B,X,DECOMP_SVD);
	
	return X;
}

/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
Mat_<double> IterativeLinearLSTriangulation(Point3d u,	//homogenous image point (u,v,1)
											Matx34d P,			//camera 1 matrix
											Point3d u1,			//homogenous image point in 2nd camera
											Matx34d P1			//camera 2 matrix
											) {
	double wi = 1, wi1 = 1;
	Mat_<double> X(4,1); 
	for (int i=0; i<10; i++) { //Hartley suggests 10 iterations at most
		Mat_<double> X_ = LinearLSTriangulation(u,P,u1,P1);
		X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;
		
		//recalculate weights
		double p2x = Mat_<double>(Mat_<double>(P).row(2)*X)(0);
		double p2x1 = Mat_<double>(Mat_<double>(P1).row(2)*X)(0);
		
		//breaking point
		if(fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON) break;
		
		wi = p2x;
		wi1 = p2x1;
		
		//reweight equations and solve
		Matx43d A((u.x*P(2,0)-P(0,0))/wi,		(u.x*P(2,1)-P(0,1))/wi,			(u.x*P(2,2)-P(0,2))/wi,		
				  (u.y*P(2,0)-P(1,0))/wi,		(u.y*P(2,1)-P(1,1))/wi,			(u.y*P(2,2)-P(1,2))/wi,		
				  (u1.x*P1(2,0)-P1(0,0))/wi1,	(u1.x*P1(2,1)-P1(0,1))/wi1,		(u1.x*P1(2,2)-P1(0,2))/wi1,	
				  (u1.y*P1(2,0)-P1(1,0))/wi1,	(u1.y*P1(2,1)-P1(1,1))/wi1,		(u1.y*P1(2,2)-P1(1,2))/wi1
				  );
		Mat_<double> B = (Mat_<double>(4,1) <<	-(u.x*P(2,3)	-P(0,3))/wi,
						  -(u.y*P(2,3)	-P(1,3))/wi,
						  -(u1.x*P1(2,3)	-P1(0,3))/wi1,
						  -(u1.y*P1(2,3)	-P1(1,3))/wi1
						  );
		
		solve(A,B,X_,DECOMP_SVD);
		X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;
	}
	return X;
}

//Triagulate points
void TriangulatePoints(const vector<Point2d>& pt_set1, 
					   const vector<Point2d>& pt_set2, 
					   const Mat& Kinv,
					   const Matx34d& P,
					   const Matx34d& P1,
					   vector<Point3d>& pointcloud,
					   vector<Point>& correspImg1Pt)
{
#ifdef __SFM__DEBUG__
	vector<double> depths;
#endif
	
	pointcloud.clear();
	correspImg1Pt.clear();
	
	cout << "Triangulating...";
	double t = getTickCount();
	unsigned int pts_size = pt_set1.size();
#pragma omp parallel for
	for (unsigned int i=0; i<pts_size; i++) {
		Point2f kp = pt_set1[i]; 
		Point3d u(kp.x,kp.y,1.0);
		Mat_<double> um = Kinv * Mat_<double>(u); 
		u = um.at<Point3d>(0);
		Point2f kp1 = pt_set2[i]; 
		Point3d u1(kp1.x,kp1.y,1.0);
		Mat_<double> um1 = Kinv * Mat_<double>(u1); 
		u1 = um1.at<Point3d>(0);
		
		Mat_<double> X = IterativeLinearLSTriangulation(u,P,u1,P1);
		
//		if(X(2) > 6 || X(2) < 0) continue;
		
#pragma omp critical
		{
			pointcloud.push_back(Point3d(X(0),X(1),X(2)));
			correspImg1Pt.push_back(pt_set1[i]);
#ifdef __SFM__DEBUG__
			depths.push_back(X(2));
#endif
		}
	}
	t = ((double)getTickCount() - t)/getTickFrequency();
	cout << "Done. ("<<pointcloud.size()<<"points, " << t <<"s)"<< endl;
	
	//show "range image"
#ifdef __SFM__DEBUG__
	{
		double minVal,maxVal;
		minMaxLoc(depths, &minVal, &maxVal);
		Mat tmp(240,320,CV_8UC3); //cvtColor(img_1_orig, tmp, CV_BGR2HSV);
		for (unsigned int i=0; i<pointcloud.size(); i++) {
			double _d = MAX(MIN((pointcloud[i].z-minVal)/(maxVal-minVal),1.0),0.0);
			circle(tmp, correspImg1Pt[i], 1, Scalar(255 * (1.0-(_d)),255,255), CV_FILLED);
		}
		cvtColor(tmp, tmp, CV_HSV2BGR);
		imshow("Depth Map", tmp);
		waitKey(0);
		destroyWindow("Depth Map");
	}	
#endif
}