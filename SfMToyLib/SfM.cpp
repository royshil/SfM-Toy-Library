/*
 * SfM.cpp
 *
 *  Created on: May 26, 2016
 *      Author: roy_shilkrot
 * 
 *  Copyright @  2016
 */

#include "SfM.h"
#include "SfMStereoUtilities.h"

#include <iostream>
#include <algorithm>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;

namespace sfmtoylib {

const float POSE_INLIERS_MINIMAL_RATIO = 0.5;

SfM::SfM() {
}

SfM::~SfM() {
}

ErrorCode SfM::runSfM() {
    if (mImages.size() <= 0) {
        cerr << "No images to work on." << endl;
        return ERROR;
    }

    //initialize intrinsics
    mIntrinsics.K = (Mat_<float>(3,3) << 700,   0, mImages[0].cols / 2,
                                           0, 700, mImages[0].rows / 2,
                                           0,   0,  1);
    mIntrinsics.Kinv = mIntrinsics.K.inv();
    mIntrinsics.distortion = Mat_<float>::zeros(1, 4);

    //First - extract features from all images
    extractFeatures();

    //Create a matching matrix between all images' features
    createFeatureMatchMatrix();

    //Find the best two views for an initial triangulation on the 3D map
    findBaselineTriangulation();

    //Lastly - add more camera views to the map
    addMoreViewsToReconstruction();

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
        string extension = x.path().extension().string();
        boost::algorithm::to_lower(extension);
        if (extension == ".jpg" or extension == ".png") {
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
    cout << "----------------- Extract Features -----------------" << endl;

    mImageFeatures.resize(mImages.size());
    for (size_t i = 0; i < mImages.size(); i++) {
        mImageFeatures[i] = mFeatureUtil.extractFeatures(mImages[i]);

        cout << "Image " << i << ": " << mImageFeatures[i].keyPoints.size() << " keypoints" << endl;
    }
}


void SfM::createFeatureMatchMatrix() {
    cout << "----------- Create Feature Match Matrix ------------" << endl;

    size_t numImages = mImages.size();
    mFeatureMatchMatrix.resize(numImages, vector<Matching>(numImages));

    for (size_t i = 0; i < numImages; i++) {
        for (size_t j = i + 1; j < numImages; j++) {
            mFeatureMatchMatrix[i][j] = mFeatureUtil.matchFeatures(mImageFeatures[i], mImageFeatures[j]);

            cout << "Match " << i << ", " << j << ": " << mFeatureMatchMatrix[i][j].size() << " matched features" << endl;
        }
    }
}


void SfM::findBaselineTriangulation() {
    cout << "----------- Find Baseline Triangulation ------------" << endl;

    cout << "--- Sort views by homography inliers" << endl;

    //maps are sorted, so the best pair is at the beginnning
    map<float, ImagePair> pairsHomographyInliers = sortViewsForBaseline();

    Matx34f Pleft  = Matx34f::eye();
    Matx34f Pright = Matx34f::eye();
    Features prunedLeft, prunedRight;
    PointCloud pointCloud;

    cout << "--- Try views in triangulation" << endl;

    //try to find the best pair, stating at the beginning
    for (auto& imagePair : pairsHomographyInliers) {
        cout << "Trying " << imagePair.second << " ratio: " << imagePair.first << endl << flush;
        size_t i = imagePair.second.left;
        size_t j = imagePair.second.right;

        cout << "---- Find camera matrices" << endl;
        //recover camera matrices (poses) from the point matching
        bool success = SfMStereoUtilities::findCameraMatricesFromMatch(
                mIntrinsics,
                mFeatureMatchMatrix[i][j],
                mImageFeatures[i], mImageFeatures[j],
                prunedLeft,        prunedRight,
                Pleft,             Pright
                );

        if (not success) {
            cerr << "stereo view could not be obtained " << imagePair.second << ", go to next pair" << endl << flush;
            continue;
        }

        float poseInliersRatio = (float)prunedLeft.keyPoints.size() / (float)mFeatureMatchMatrix[i][j].size();

        cout << "pose inliers ratio " << poseInliersRatio << endl;

        if (poseInliersRatio < POSE_INLIERS_MINIMAL_RATIO) {
            cerr << "insufficient pose inliers." << endl;
            continue;
        }

        cout << "---- Triangulate from stereo views" << endl;
        success = SfMStereoUtilities::triangulateViews(
                mIntrinsics,
                imagePair.second,
                mFeatureMatchMatrix[i][j],
                mImageFeatures[i], mImageFeatures[j],
                Pleft, Pright,
                pointCloud
                );

       if (not success) {
           cerr << "could not triangulate: " << imagePair.second << endl << flush;
           continue;
       }

       mReconstructionCloud = pointCloud;

       adjustCurrentBundle();

       break;
    }
}


void SfM::adjustCurrentBundle() {
    cv::detail::BundleAdjusterReproj ba;
}


map<float, ImagePair> SfM::sortViewsForBaseline() {
    cout << "---------- Find Views Homography Inliers -----------" << endl;

    //sort pairwise matches to find the lowest Homography inliers [Snavely07 4.2]
    map<float, ImagePair> matchesSizes;
    size_t numImages = mImages.size();
    for (size_t i = 0; i < numImages; i++) {
        for (size_t j = i + 1; j < numImages; j++) {
            if (mFeatureMatchMatrix[i][j].size() < 100) {
                //Not enough points in matching
                matchesSizes[1.0] = {i, j};
                continue;
            }

            //Find number of homography inliers
            int numInliers = SfMStereoUtilities::findHomographyInliers(
                    mImageFeatures[i],
                    mImageFeatures[j],
                    mFeatureMatchMatrix[i][j]);
            float inliersRatio = (float)numInliers / (float)(mFeatureMatchMatrix[i][j].size());
            matchesSizes[inliersRatio] = {i, j};

            cout << "Homography inliers ratio: " << i << ", " << j << " " << inliersRatio << endl;
        }
    }

    return matchesSizes;
}

void SfM::addMoreViewsToReconstruction() {
    cout << "----------------- Add More Views ------------------" << endl;


}

} /* namespace sfmtoylib */
