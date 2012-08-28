/*
 *  Deprecated.cpp
 *  EyeRingOpenCV
 *
 *  Created by Roy Shilkrot on 12/24/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */


/** @function main */
int _main( int argc, char** argv )
{
	if( argc != 3 )
		//{ readme(); return -1; }
	{
		cout<<"No arguments. Using default files.";
		argv = new char*[2];
		argv[1]="Stereo-S2/D1.jpg";
		argv[2]="Stereo-S2/D2.jpg";
	}
	
	Mat img_1 = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
	Mat img_2 = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );
	
	if( !img_1.data || !img_2.data )
	{ std::cout<< " --(!) Error reading images " << std::endl; return -1; }
	
	
	Mat cam_matrix,distortion_coeff;
	FileStorage fs;
	//fs.open("Camera_Calibration.txt",FileStorage::READ);
	fs.open("Calibration_S2.txt",FileStorage::READ);
	fs["camera_matrix"]>>cam_matrix;
	cout<<"Camera Matrix:"<<endl<<cam_matrix<<endl;
	fs["distortion_coefficients"]>>distortion_coeff;
	cout<<"Distortion coefficients:"<<endl<<distortion_coeff<<endl;
	
	////**Undistort image using calibration matrices
	//Mat temp_calib_1,temp_calib_2;
	//undistort(img_1,temp_calib_1,cam_matrix,distortion_coeff);
	//undistort(img_2,temp_calib_2,cam_matrix,distortion_coeff);
	//img_1=temp_calib_1;
	//img_2=temp_calib_2;
	
	
	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;
	
	SurfFeatureDetector detector( minHessian );
	
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	
	
	detector.detect( img_1, keypoints_1 );
	detector.detect( img_2, keypoints_2 );
	
	//-- Step 2: Calculate descriptors (feature vectors)
	SurfDescriptorExtractor extractor;
	
	Mat descriptors_1, descriptors_2;
	
	extractor.compute( img_1, keypoints_1, descriptors_1 );
	extractor.compute( img_2, keypoints_2, descriptors_2 );
	
	//-- Step 3: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match( descriptors_1, descriptors_2, matches );
	
	double max_dist = 0; double min_dist = 100;
	//-- Quick calculation of max and min distances between keypoints
	for( int i = 0; i < descriptors_1.rows; i++ )
	{ double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}
	
	printf("-- Max dist : %f \n", max_dist );
	printf("-- Min dist : %f \n", min_dist );
	
	//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
	//-- PS.- radiusMatch can also be used here.
	std::vector< DMatch > good_matches;
	
	for( int i = 0; i < descriptors_1.rows; i++ )
	{ if( matches[i].distance < 3*min_dist )
    { good_matches.push_back( matches[i]); }
	}
	
	//-- Draw only "good" matches
	Mat img_matches;
	drawMatches( img_1, keypoints_1, img_2, keypoints_2,
				good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	
	//-- Show detected matches
	imshow( "Good Matches", img_matches );
	
	
	Mat left_img_keypoints, right_img_keypoints;
	Mat left_col1,left_col2,right_col1,right_col2;
	left_img_keypoints= Mat(good_matches.size(),3,CV_64F);
	right_img_keypoints= Mat(good_matches.size(),3,CV_64F);
	
	for( int i = 0; i < good_matches.size(); i++ )
	{ printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx );
		left_img_keypoints.at<double>(i,0) = keypoints_1[good_matches[i].queryIdx].pt.x;
		left_img_keypoints.at<double>(i,1) = keypoints_1[good_matches[i].queryIdx].pt.y;
		left_img_keypoints.at<double>(i,2) = 1.0;
		
		right_img_keypoints.at<double>(i,0) = keypoints_2[good_matches[i].trainIdx].pt.x;
		right_img_keypoints.at<double>(i,1) = keypoints_2[good_matches[i].trainIdx].pt.y;
		right_img_keypoints.at<double>(i,2) = 1.0;
		//left_col1.push_back(keypoints_1[good_matches[i].queryIdx].pt.x);
		//left_col1.push_back(keypoints_1[good_matches[i].queryIdx].pt.y);
		//left_img_keypoints.push_back(keypoints_1[good_matches[i].queryIdx].pt);
		//right_img_keypoints.push_back(keypoints_2[good_matches[i].trainIdx].pt);
		
	}
	
	//cout<<"Homogenous1:"<<endl<<left_img_keypoints<<endl;
	//convertPointsToHomogeneous(left_img_keypoints,left_img_keypoints);
	//convertPointsToHomogeneous(right_img_keypoints,right_img_keypoints);
	
	
	Mat RT_left,RT_right;
	double r1[]={1,0,0,50,0,1,0,0,0,0,1,0};
	double r2[]={1,0,0,-50,0,1,0,0,0,0,1,0};
	RT_left=Mat(3,4,CV_64F,r1);
	RT_right=Mat(3,4,CV_64F,r2);
	
	//double r1[]={1,0,0,0,1,0,0,0,1};
	//double r2[]={1,0,0,0,1,0,0,0,1};
	//RT_left=Mat(3,3,CV_64F,r1);
	//RT_right=Mat(3,3,CV_64F,r2);
	
	Mat left_ART_matrix = cam_matrix*RT_left;
	Mat right_ART_matrix = cam_matrix*RT_right;
	//cout<<"A*RT:"<<endl<<left_ART_matrix<<endl;
	
	Mat left_world_coords=Mat(4,1,CV_64F);
	Mat right_world_coords=Mat(4,1,CV_64F);
	//  double elem1[]={1,2,3,4,5,6,7,8,9,10,11,12};
	//  double elem2[]={2,5,7};
	//Mat temp1 = Mat(3,4,CV_64F,elem1);
	//Mat temp2 = Mat(3,1,CV_64F,elem2);
	//solve(temp1.t()*temp1,temp1.t()*temp2,coords,DECOMP_SVD);
	Mat left_cam_xyz  = (Mat_<double>(3,1) << -50,0,0); //Camera positions
	Mat right_cam_xyz  = (Mat_<double>(3,1) << 50,0,0);
	
	Mat World_Point_array;
	double max_z=0;
	double min_z=10000000000;
	
	for( int i = 0; i < good_matches.size(); i++ ){
		
		Mat left_image_xyz = left_img_keypoints.row(i).t();
		Mat right_image_xyz = right_img_keypoints.row(i).t();
		solve(left_ART_matrix.t()*left_ART_matrix,left_ART_matrix.t()*left_image_xyz,left_world_coords,DECOMP_SVD);
		solve(right_ART_matrix.t()*right_ART_matrix,right_ART_matrix.t()*right_image_xyz,right_world_coords,DECOMP_SVD);
		//solve(ART_matrix,left_img_keypoints(Range(0,1),Range(0,1)).t(),coords,DECOMP_SVD);
		//solve(left_img_keypoints(Range(0,1),Range(0,1)).t(),ART_matrix,coords,DECOMP_SVD);
		double tempscale1 = left_world_coords.at<double>(3,0);
		double tempscale2 = right_world_coords.at<double>(3,0);
		left_world_coords=left_world_coords/tempscale1;
		right_world_coords=right_world_coords/tempscale2;
		
		/*cout<<"Left :"<<left_world_coords<<endl;
		 cout<<"Right:"<<right_world_coords<<endl;*/
		
		double leftelem[]={left_world_coords.at<double>(0,0),left_world_coords.at<double>(1,0),left_world_coords.at<double>(2,0)};
		double rightelem[]={right_world_coords.at<double>(0,0),right_world_coords.at<double>(1,0),right_world_coords.at<double>(2,0)};
		Mat left_3D_coords=Mat(3,1,CV_64F,leftelem);
		Mat right_3D_coords=Mat(3,1,CV_64F,rightelem);
		
		//cout<<"Left :"<<left_3D_coords<<endl;
		//cout<<"Right:"<<right_3D_coords<<endl;
		
		Mat u = left_3D_coords-left_cam_xyz; //Compute direction vectors of the rays
		Mat v = right_3D_coords-right_cam_xyz;
		Mat w = left_cam_xyz-right_cam_xyz;
		
		double a = u.dot(u);
		double b = u.dot(v);
		double c = v.dot(v);
		double d = u.dot(w);
		double e = v.dot(w);
		
		double s_left = ((b*e)-(c*d))/((a*c)-(b*b));
		double s_right = ((a*e)-(b*d))/((a*c)-(b*b));
		
		Mat point_left = left_cam_xyz + (s_left*u);
		Mat point_right = right_cam_xyz + (s_right*v);
		Mat Match_Point_3D = (point_left+point_right)/2;
		Mat point_diff_vec = ((left_cam_xyz-right_cam_xyz)+(s_left*u)-(s_right*v));
		/*double point_distance = pow(point_diff_vec.dot(point_diff_vec),0.5);
		 cout<<"P1:"<<point_left<<endl;
		 cout<<"P2:"<<point_right<<endl;
		 cout<<"Distance:"<<point_distance<<endl;*/
		
		Match_Point_3D=Match_Point_3D.t();
		World_Point_array.push_back(Match_Point_3D.row(0));
		
		if (abs(Match_Point_3D.at<double>(0,2))>max_z){
			max_z=abs(Match_Point_3D.at<double>(0,2));
		}
		if (abs(Match_Point_3D.at<double>(0,2))<min_z){
			min_z=abs(Match_Point_3D.at<double>(0,2));
		}
		
	}
	
	cout<<"Min Z: "<<min_z<<endl;
	cout<<"Max Z: "<<max_z<<endl;
	
	Mat img_depth=img_1;
	
	
	for(unsigned int i = 0; i < good_matches.size(); i++ ){
		double _d = (World_Point_array.at<double>(i,2)-min_z)/(max_z-min_z);
		Scalar color = Scalar(1.0-_d*255);
		circle(img_depth,Point2d(left_img_keypoints.at<double>(i,0),left_img_keypoints.at<double>(i,1)),5,color,-1);
	}
	
	imshow("Temp",img_depth);
	
	//cout<<"Mult:"<<temp1*coords.t()<<endl;
	
	
	//Mat Rot = Mat::eye(Size(3,3),CV_64F);
	//double translation1[]={100,0,0};
	//Mat t = Mat(3,1,CV_64F,translation1);
	//Mat World_coords = Mat(good_matches.size(),3,CV_64F);
	//
	//for( int i = 0; i < good_matches.size(); i++ ){
	
	// Mat r1 = Rot.row(0);
	// Mat r3 = Rot.row(2);
	// Mat row = right_img_keypoints.row(i);
	// Mat x3 = ((r1 - (row.col(1)*r3))*t)/((r1 - (row.col(1)*r3))*row.t());
	// Mat x3_2 = ((r1 - (row.col(2)*r3))*t)/((r1 - (row.col(2)*r3))*row.t());
	// cout<<x3<<","<<x3_2<<endl;
	
	
	
	//}
	//cout<<left_img_keypoints.size().height<<","<<left_img_keypoints.size().width<<endl;
	
	
	waitKey(0);
	int a=0;
	cin>>a;
	return 0;
}

/** @function readme */
void readme()
{ std::cout << " Usage: ./SURF_FlannMatcher <img1> <img2>" << std::endl; }



int __main(int argc, char** argv) { //randomized test
	//---- test triangulation ----
	//randomize 3D points
	vector<Matx41d> points3d;
	RNG rng;
	for (int i=0; i<100; i++) {
		Matx41d p;
		p(0) = rng.uniform(-220.0, 220.0);
		p(1) = rng.uniform(-140.0, 140.0);
		p(2) = rng.uniform(250.0, 2150.0);
		p(3) = 1.0;
		points3d.push_back(p);
	}
	//project onto 2 cameras using P=[R|t] matrices
	Matx34d P(1,0,0,-50,
			  0,1,0,0,
			  0,0,1,0);
	Matx34d P1(1,0,0,50,
			   0,1,0,0,
			   0,0,1,0);
	vector<Point3d> points2d,points2d1;
	Mat_<Vec3b> img(1000,1000);
	for (unsigned int i=0; i<points3d.size(); i++) {
		Matx41d X = points3d[i];
		Matx31d a;
		
		a = P * X;
		Point3d x(a(0)/a(2),a(1)/a(2),1.0);
		points2d.push_back(x);
		circle(img,Point(x.x*1000.0+500.0,x.y*1000.0+500.0),2,Scalar(255.0*(1.0-a(2)/2150.0)),CV_FILLED);
		
		a = P1 * X;
		x = Point3d(a(0)/a(2),a(1)/a(2),1.0);
		points2d1.push_back(x);
		circle(img,Point(x.x*1000.0+500.0,x.y*1000.0+500.0),2,Scalar(0,255.0*(1.0-a(2)/2150.0)),CV_FILLED);
	}
	imshow("tmp",img); waitKey(0);
	
	//re-triangulate and calculate error
	vector<Matx31d> triangulated;
	for (unsigned int i=0; i<points3d.size(); i++) {
		Mat_<double> res = IterativeLinearLSTriangulation(points2d[i],P,points2d1[i],P1);
		Point3d o(points3d[i](0),points3d[i](1),points3d[i](2));
		Vec4d tri_orig(res(0),res(1),res(2),res(3));
		//		Point3d tri(res(0)/res(3),res(1)/res(3),res(2)/res(3));
		Point3d tri(res(0),res(1),res(2));
		cout << "Original " << o << 
		", Triangulated ("<<tri_orig[0]<<","<<tri_orig[1]<<","<<tri_orig[2]<<","<<tri_orig[3]<<")" << tri << endl;
		
		Matx41d X = tri_orig; ///tri_orig[3]; //homogeneous
		Matx31d a;
		
		a = P * X;
		Point3d x(a(0)/a(2),a(1)/a(2),1.0);
		circle(img,Point(x.x*1000.0+500.0,x.y*1000.0+500.0),2,Scalar(0,0,255.0*(1.0-a(2))),CV_FILLED);
		
		a = P1 * X;
		x = Point3d(a(0)/a(2),a(1)/a(2),1.0);
		circle(img,Point(x.x*1000.0+500.0,x.y*1000.0+500.0),2,Scalar(255,0,255.0*(1.0-a(2))),CV_FILLED);		
	}
	imshow("tmp", img); waitKey(0);
	return 0;
}