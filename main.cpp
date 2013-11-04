/*
 *  main.cpp
 *  ExploringSfMWithOpenCV
 *
 *  Created by Roy Shilkrot on 4/27/12.
 *  Copyright 2012 MIT. All rights reserved.
 *
 */

#include <iostream>
#include <string.h>

#include "Distance.h"
#include "MultiCameraPnP.h"

#ifdef HAVE_GUI
#include <QApplication>
#include <QGLFormat>

#include "ViewerInterface.h"
#endif

using namespace std;

#ifdef HAVE_OPENCV_GPU
#include <opencv2/gpu/gpu.hpp>
#endif

#ifdef HAVE_GUI
//------------------------------ Using Qt GUI ------------------------------
int main(int argc, char** argv) { //test with real photos
	// Read command lines arguments.
	QApplication application(argc,argv);

	QGLFormat glFormat;
	glFormat.setVersion( 3, 2 );
	glFormat.setProfile( QGLFormat::CoreProfile ); // Requires >=Qt-4.8.0
	//    glFormat.setSampleBuffers( true );
	QGLFormat::setDefaultFormat(glFormat);

	// Instantiate the viewer.
	ViewerInterface viewer;

	viewer.setWindowTitle("SfM-Toy-Library UI");
	
	// Make the viewer window visible on screen.
	viewer.show();

	// Run main loop.
	return application.exec();
}

#else
std::vector<cv::Mat> images;
std::vector<std::string> images_names;


void open_imgs_dir(char* dir_name, std::vector<cv::Mat>& images, std::vector<std::string>& images_names);

//---------------------------- Using command-line ----------------------------

int main(int argc, char** argv) {
	if (argc < 2) {
		cerr << "USAGE: " << argv[0] << " <path_to_images> [use rich features (RICH/OF) = RICH] [use GPU (GPU/CPU) = GPU] [downscale factor = 1.0]" << endl;
		return 0;
	}
	
	double downscale_factor = 1.0;
	if(argc >= 5)
		downscale_factor = atof(argv[4]);

	open_imgs_dir(argv[1],images,images_names,downscale_factor);
	if(images.size() == 0) { 
		cerr << "can't get image files" << endl;
		return 1;
	}

	
	cv::Ptr<MultiCameraPnP> distance = new MultiCameraPnP(images,images_names,string(argv[1]));
	if(argc < 3)
		distance->use_rich_features = true;
	else
		distance->use_rich_features = (strcmp(argv[2], "RICH") == 0);
	
#ifdef HAVE_OPENCV_GPU
	if(argc < 4)
		distance->use_gpu = (cv::gpu::getCudaEnabledDeviceCount() > 0);
	else
		distance->use_gpu = (strcmp(argv[3], "GPU") == 0);
#else
	distance->use_gpu = false;
#endif
	
	cv::Ptr<VisualizerListener> visualizerListener = new VisualizerListener; //with ref-count
	distance->attach(visualizerListener);
	RunVisualizationThread();

	distance->RecoverDepthFromImages();

	//TODO: save point cloud and cameras to file
}
#endif
