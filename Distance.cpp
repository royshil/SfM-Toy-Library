#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/video/tracking.hpp"
#include <fstream>
#include "dirent.h"


using namespace std;


using namespace cv;

#include "Distance.h"

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

#include "MultiCameraPnP.h"



std::vector<cv::Mat> images;
std::vector<std::string> images_names;

bool hasEnding (std::string const &fullString, std::string const &ending)
{
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

void open_imgs_dir(char* dir_name) {
	if (dir_name == NULL) {
		return;
	}
	std::vector<std::string> files_;
	DIR *dp;
	struct dirent *ep;     
	dp = opendir (dir_name);
	
	if (dp != NULL)
	{
		while (ep = readdir (dp)) {
			if (ep->d_name[0] != '.')
				files_.push_back(ep->d_name);
		}
		
		(void) closedir (dp);
	}
	else {
		std::cerr << ("Couldn't open the directory");
		return;
	}
	for (unsigned int i=0; i<files_.size(); i++) {
		if (files_[i][0] == '.' || !(hasEnding(files_[i],"jpg")||hasEnding(files_[i],"png"))) {
			continue;
		}
		cv::Mat m_ = cv::imread(std::string(dir_name) + "/" + files_[i]);
		images_names.push_back(files_[i]);
		images.push_back(m_);
	}
}

int main(int argc, char** argv) {
	if (argc != 2) {
		cerr << "USAGE: " << argv[0] << " <path_to_images>" << endl;
		return 0;
	}
	
	open_imgs_dir(argv[1]);
	
	cv::Ptr<IDistance> distance = new MultiCameraPnP(images,images_names);
	distance->RecoverDepthFromImages();
	
	RunVisualization(distance->getPointCloud(), distance->getPointCloudRGB());
}
#endif