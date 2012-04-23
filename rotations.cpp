/*
 *  rotations.cpp
 *  SfMToyExample
 *
 *  Created by Roy Shilkrot on 4/17/12.
 *  Copyright 2012 MIT. All rights reserved.
 *
 */

#include "opencv2/opencv.hpp"
using namespace cv;

#include <iostream>
using namespace std;

#define DEG2RAD(DEG) ((DEG)*((CV_PI)/(180.0)))

int main(int argc, char** argv) {
	
	vector<Point3d> X;
	X.push_back(Point3d(10,23,-7));
	X.push_back(Point3d(1,13,7));
	X.push_back(Point3d(14,2,-17));
	X.push_back(Point3d(4,21,1));
	X.push_back(Point3d(9,5,-1));
	Mat Xm = Mat(X).reshape(1).t();
	
	double thetha = DEG2RAD(2.0);
	double c = cos(thetha);
	double s = sin(thetha);
//	Mat_<double> R = (Mat_<double>(3,3) << 1,0,0,
//										  0,c,-s,
//										  0,s,c);
	Mat_<double> R = (Mat_<double>(3,3) << -0.9958804223676606, -0.0111691307288813, 0.08998574811363567,
											0.01152533877557442, -0.9999276644222462, 0.003439838539782635,
										  -0.08994081893621918, -0.004462784089858307, -0.9959371127974144);
//	Mat_<double> R1 = (Mat_<double>(3,3) << 1,0,0,
//										   0,c,-s,
//										   0,s,c);
	Mat_<double> R1 = (Mat_<double>(3,3) << -0.993334171865709, 0.01890711763861502, 0.1137090317717334,
					  -0.02100959614581286, -0.99962923823526, -0.01732001544410723,
					   -0.1133394012409425, 0.01959354403352907, -0.9933631124410401);
	Mat_<double> Rm1 = R.inv(), R1m1 = R1.inv();
	
	cout << "R " << R << endl;
	cout << "R*(-I) " << R*(-Mat_<double>::eye(3,3)) << endl;
	cout << "R-1 " << Rm1 << endl;
	cout << "R1 " << R1 << endl;
	cout << "R1-1 " << R1m1 << endl;
	cout << "R*R1 " << R*(-Mat_<double>::eye(3,3))*R1*(-Mat_<double>::eye(3,3)) << endl;
	cout << "R1*R " << R1*R << endl;
	cout << "R1-1*R " << R1m1*R << endl;
	
	cout << "X " << X << endl;
	cout << "R * X " << (R * Xm).t() << endl;
	cout << "R1 * R * X " << (R1 * R * Xm).t() << endl;
	cout << "R * R1 * X " << (R * R1 * Xm).t() << endl;
	cout << "R-1 * R * X " << (Rm1 * R * Xm).t() << endl;

}

