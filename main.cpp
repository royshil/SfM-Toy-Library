/*
 *  main.cpp
 *  ExploringSfMWithOpenCV
 *
 *  Created by Roy Shilkrot on 4/27/12.
 *  Copyright 2012 MIT. All rights reserved.
 *
 */

#include <iostream>

#include "Distance.h"
#include "MultiCameraPnP.h"
#include "Visualization.h"

using namespace std;

std::vector<cv::Mat> images;
std::vector<std::string> images_names;


void open_imgs_dir(char* dir_name, std::vector<cv::Mat>& images, std::vector<std::string>& images_names);

#ifndef NO_FLTK
//------------------------------ Using FLTK GUI ------------------------------
int runUI(int argc, char** argv);

int main(int argc, char** argv) { //test with real photos
	runUI(argc, argv);

	destroyAllWindows();
	
//	RunVisualization(pointcloud, img_1_orig, img_2_orig, correspImg1Pt);
	
    return 0;
}

#else
//---------------------------- Using command-line ----------------------------

int main(int argc, char** argv) {
	if (argc != 2) {
		cerr << "USAGE: " << argv[0] << " <path_to_images>" << endl;
		return 0;
	}
	
	open_imgs_dir(argv[1],images,images_names);
	if(images.size() == 0) { 
		cerr << "can't get image files" << endl;
		return 1;
	}
	
	cv::Ptr<MultiCameraPnP> distance = new MultiCameraPnP(images,images_names,string(argv[1]));
	distance->RecoverDepthFromImages();
	
	RunVisualization(distance->getPointCloud(), 
					 distance->getPointCloudRGB(),
					 distance->getPointCloudBeforeBA(),
					 distance->getPointCloudRGBBeforeBA()
					 );
}
#endif