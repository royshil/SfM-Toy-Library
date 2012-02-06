#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/video/tracking.hpp"
#include <fstream>

using namespace std;


using namespace cv;

#include "Distance.h"


int runUI(int argc, char** argv);

int main(int argc, char** argv) { //test with real photos
	runUI(argc, argv);

	destroyAllWindows();
	
//	RunVisualization(pointcloud, img_1_orig, img_2_orig, correspImg1Pt);
	
    return 0;

}