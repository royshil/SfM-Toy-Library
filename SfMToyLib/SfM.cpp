/*
 * SfM.cpp
 *
 *  Created on: May 26, 2016
 *      Author: roy_shilkrot
 * 
 *  Copyright @  2016
 */

#include "SfM.h"

#include <iostream>

#include <opencv2/highgui/highgui.hpp>

#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;

namespace sfmtoylib {

SfM::SfM() {
    // TODO Auto-generated constructor stub

}

SfM::~SfM() {
    // TODO Auto-generated destructor stub
}

ErrorCode SfM::runSfM() {
    if (mImages.size() <= 0) {
        cerr << "No images to work on." << endl;
        return ERROR;
    }

    extractFeatures();
    createFeatureMatchMatrix();
    findBaselineTriangulation();
    adjustCurrentBundle();

    return OKAY;
}

bool SfM::setImagesDirectory(const std::string directoryPath) {
    using namespace boost::filesystem;

    path dirPath(directoryPath);
    if (not exists(dirPath) or not is_directory(dirPath)) {
        cerr << "Cannot open directory: " << directoryPath << endl;
        return false;
    }

    for (directory_entry& x : directory_iterator(dirPath)) {
        const string extension = x.path().extension().string();
        if (extension == "jpg" or extension == "png") {
            mImageFilenames.push_back(x.path().string());
        }
    }

    if (mImageFilenames.size() <= 0) {
        cerr << "Unable to find valid files in images directory." << endl;
        return false;
    }

    cout << "Found " << mImageFilenames.size() << " image files in directory." << endl;

    for (auto& imageFilename : mImageFilenames) {
        mImages.push_back(imread(imageFilename));

        if (mImages.back().empty()) {
            cerr << "Unable to read image from file: " << imageFilename << endl;
            return false;
        }
    }

    return true;
}

void SfM::extractFeatures() {
    mImageFeatures.reserve(mImages.size());
    for (size_t i = 0; i < mImages.size(); i++) {
        mFeatureUtilities.extractFeatures(mImages[i], mImageFeatures[i]);
    }
}

void SfM::createFeatureMatchMatrix() {
}

void SfM::findBaselineTriangulation() {
}

void SfM::adjustCurrentBundle() {
}

} /* namespace sfmtoylib */
