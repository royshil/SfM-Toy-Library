/*
 *  OFFeatureMatcher.cpp
 *  ExploringSfMWithOpenCV
 *
 *  Created by Roy Shilkrot on 5/17/12.
 *  Copyright 2012 MIT. All rights reserved.
 *
 */

#include "OFFeatureMatcher.h"
#include <opencv2/video/video.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#ifdef __SFM__DEBUG__
#include <opencv2/highgui/highgui.hpp>
#endif

using namespace std;
using namespace cv;

//c'tor
OFFeatureMatcher::OFFeatureMatcher(std::vector<cv::Mat>& imgs_, 
								std::vector<std::vector<cv::KeyPoint> >& imgpts_) :
imgpts(imgpts_), imgs(imgs_)
{
	//detect keypoints for all images
	FastFeatureDetector ffd;
	ffd.detect(imgs, imgpts);
}

void OFFeatureMatcher::MatchFeatures(int idx_i, int idx_j, vector<DMatch>* matches) {	
	vector<Point2f> i_pts; 
	KeyPointsToPoints(imgpts[idx_i],i_pts);
	
	vector<Point2f> j_pts(i_pts.size());
	
	// making sure images are grayscale
	Mat prevgray,gray;
	if (imgs[idx_i].channels() == 3) {
		cvtColor(imgs[idx_i],prevgray,CV_RGB2GRAY);
		cvtColor(imgs[idx_j],gray,CV_RGB2GRAY);
	} else {
		prevgray = imgs[idx_i];
		gray = imgs[idx_j];
	}
						  
	vector<uchar> vstatus; vector<float> verror;
	calcOpticalFlowPyrLK(prevgray, gray, i_pts, j_pts, vstatus, verror);
	
	for (unsigned int i=0; i<vstatus.size(); i++) {
#ifdef __SFM__DEBUG__
		if (vstatus[i]) {
			if (i%10==0 && verror[i] < 20.0) { //prune some matches for display purposes
				vstatus[i] = 1;
			} else {
				vstatus[i] = 0;
			}
#endif

			matches->push_back(DMatch(i,i,1.0));
		}
	}
	
	PointsToKeyPoints(j_pts,imgpts[idx_j]);
	
#ifdef __SFM__DEBUG__
	{
		// draw flow field
		Mat img_matches; cvtColor(imgs[idx_i],img_matches,CV_GRAY2BGR);
		drawArrows(img_matches, i_pts, j_pts, vstatus, Scalar(0,255));
		imshow( "Flow Field", img_matches );
		waitKey(100);
		destroyWindow("Flow Field");
	}
#endif
}