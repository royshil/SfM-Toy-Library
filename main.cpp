/*
 *  main.cpp
 *  SfMToyUI
 *
 *  Created by Roy Shilkrot on 4/27/12.
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2013 Roy Shilkrot
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *
 */

#include <iostream>
#include <string.h>

#include "SfMToyLib/SfM.h"

#ifdef HAVE_GUI
#include <QApplication>
#include <QGLFormat>

#include "ViewerInterface.h"
#endif

using namespace std;
using namespace sfmtoylib;

#ifdef HAVE_OPENCV_GPU
#include <opencv2/gpu/gpu.hpp>
#endif

#ifdef HAVE_GUI
//------------------------------ Using Qt GUI ------------------------------
int main(int argc, char** argv) { //test with real photos
	// Read command lines arguments.
	QApplication application(argc,argv);

	// Instantiate the viewer.
	ViewerInterface viewer;

	viewer.setWindowTitle("SfM-Toy-Library UI");
	
	// Make the viewer window visible on screen.
	viewer.show();

	// Run main loop.
	return application.exec();
}

#else

//---------------------------- Using command-line ----------------------------

int main(int argc, char** argv) {
	if (argc < 2) {
		cerr << "USAGE: " << argv[0] << " <path_to_images> [downscale factor = 1.0]" << endl;
		return 0;
	}
	
	double downscale_factor = 1.0;
	if(argc >= 5)
		downscale_factor = atof(argv[4]);

	SfM sfm;
	sfm.setImagesDirectory(argv[1]);
	sfm.runSfM();

	//save point cloud and cameras to file
	sfm.saveCloudAndCamerasToPLY("output.ply");
}
#endif
