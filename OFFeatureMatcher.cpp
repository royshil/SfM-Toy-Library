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
#include <omp.h>
#include <sstream>
#endif

#include <set>

using namespace std;
using namespace cv;

//c'tor
OFFeatureMatcher::OFFeatureMatcher(std::vector<cv::Mat>& imgs_, 
								std::vector<std::vector<cv::KeyPoint> >& imgpts_) :
imgpts(imgpts_), imgs(imgs_)
{
	//detect keypoints for all images
	FastFeatureDetector ffd;
//	DenseFeatureDetector ffd;
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

	double thresh = 1.0;
	std::set<int> found_in_imgpts_j;
	for (unsigned int i=0; i<vstatus.size(); i++) {
		if (vstatus[i] && verror[i] < 9.0) {
#ifdef __SFM__DEBUG__
			if (i%3==0) { //prune some matches for display purposes
				vstatus[i] = 1;
			} else {
				vstatus[i] = 0;
			}
#endif


			//TODO must support mutual-matching, cross-matching or ratio test
			//or use patches for matching
			bool found = false;
			for(int j=0;j<imgpts[idx_j].size() && !found;j++) {
				if (found_in_imgpts_j.find(j) == found_in_imgpts_j.end()) {
					double _d = norm(j_pts[i]-imgpts[idx_j][j].pt);
					if(_d < thresh) {
						matches->push_back(DMatch(i,j,_d));
						found_in_imgpts_j.insert(j);
						found = true;
					}
				}
			}
		} else {
			vstatus[i] = 0;
		}
	}
	
	//vector<KeyPoint> tmpkpts;
	//PointsToKeyPoints(j_pts,tmpkpts);

	//
	//for(int i=0;i<tmpkpts.size();i++) {
	//	
	//}

#if 0
#ifdef __SFM__DEBUG__
	{
		// draw flow field
		Mat img_matches; cvtColor(imgs[idx_i],img_matches,CV_GRAY2BGR);
		drawArrows(img_matches, i_pts, j_pts, vstatus, verror, Scalar(0,255));
		stringstream ss; ss << "flow field " << omp_get_thread_num();
		imshow( ss.str(), img_matches );
		waitKey(500);
		destroyWindow(ss.str());
	}
#endif
#endif
}