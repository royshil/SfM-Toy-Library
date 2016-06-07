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
#include <opencv2/calib3d/calib3d.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;

namespace sfmtoylib {

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

    mCameraPoses.resize(mImages.size());

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

        Mat outImage;
        drawMatches(mImages[i], prunedLeft.keyPoints,
                mImages[j], prunedRight.keyPoints,
                GetAlignedMatching(prunedLeft.keyPoints.size()),
                outImage);
        imshow("outimage", outImage);
        waitKey(0);

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
       mCameraPoses[i] = Pleft;
       mCameraPoses[j] = Pright;
       mDoneViews.insert(i);
       mDoneViews.insert(j);
       mGoodViews.insert(i);
       mGoodViews.insert(j);

       adjustCurrentBundle();

       break;
    }
}


void SfM::adjustCurrentBundle() {
    SfMBundleAdjustmentUtils::adjustBundle(
            mReconstructionCloud,
            mCameraPoses,
            mIntrinsics,
            mImageFeatures);
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

    while (mDoneViews.size() != mImages.size()) {
        //Find the best view to add according to the largest number of 2D-3D corresponding points
        Images2D3DMatches matches2D3D = find2D3DMatches();

        size_t bestView;
        size_t bestNumMatches = 0;
        for (const auto& match2D3D : matches2D3D) {
            if (match2D3D.second.points2D.size() > bestNumMatches) {
                bestView       = match2D3D.first;
                bestNumMatches = match2D3D.second.points2D.size();
            }
        }
        cout << "Best view " << bestView << " has " << matches2D3D[bestView].points2D.size() << " matches" << endl;

        mDoneViews.insert(bestView);

        Matx34f newCameraPose;
        bool success = SfMStereoUtilities::findCameraPoseFrom2D3DMatch(
                mIntrinsics,
                matches2D3D[bestView],
                newCameraPose);

        if (not success) {
            cerr << "Cannot recover camera pose for view " << bestView << endl;
            continue;
        }

        mCameraPoses[bestView] = newCameraPose;

        cout << "view " << bestView << endl << newCameraPose << endl;

        for (const int goodView : mGoodViews) {
            //since match matrix is upper-tringular non symmetric - use lower index as left
            size_t leftViewIdx  = (goodView < bestView) ? goodView : bestView;
            size_t rightViewIdx = (goodView < bestView) ? bestView : goodView;


            PointCloud pointCloud;
            success = SfMStereoUtilities::triangulateViews(
                    mIntrinsics,
                    { leftViewIdx, rightViewIdx },
                    mFeatureMatchMatrix[leftViewIdx][rightViewIdx],
                    mImageFeatures[leftViewIdx],
                    mImageFeatures[rightViewIdx],
                    mCameraPoses[leftViewIdx],
                    mCameraPoses[rightViewIdx],
                    pointCloud
                    );

            if (success) {
                cout << "Triangulate " << leftViewIdx << " and " << rightViewIdx << " adding: " << pointCloud.size() << endl;
                mergeNewPointCloud(pointCloud);
            } else {
                cerr << "Failed to triangulate " << leftViewIdx << " and " << rightViewIdx << endl;
            }
        }

        mGoodViews.insert(bestView);
    }
}

SfM::Images2D3DMatches SfM::find2D3DMatches() {
    Images2D3DMatches matches;

    //scan all not-done views
    for (size_t viewIdx = 0; viewIdx < mImages.size(); viewIdx++) {
        if (mDoneViews.find(viewIdx) != mDoneViews.end()) {
            continue; //skip done views
        }

//        cout << "work " << viewIdx << endl;

        Image2D3DMatch match2D3D;

        //scan all cloud 3D points
        for (const Point3DInMap& cloudPoint : mReconstructionCloud) {
//            cout << "3d pt " << cloudPoint.p << endl;

            //scan all originating views for that 3D point
            for (const auto& origViewAndPoint : cloudPoint.originatingViews) {
                //check for 2D-2D matching via the match matrix
                int originatingViewIndex        = origViewAndPoint.first;
                int originatingViewFeatureIndex = origViewAndPoint.second;

//                cout << "orig view idx " << originatingViewIndex << " feature idx " << originatingViewFeatureIndex << endl;

                //match matrix is upper-triangular (not symmetric) so the left index must be the smaller one
                int leftViewIdx  = (originatingViewIndex < viewIdx) ? originatingViewIndex : viewIdx;
                int rightViewIdx = (originatingViewIndex < viewIdx) ? viewIdx : originatingViewIndex;

                //scan all 2D-2D matches between originating view and new view
                for (const DMatch& m : mFeatureMatchMatrix[leftViewIdx][rightViewIdx]) {
                    int matched2DPointInNewView = -1;
                    if (originatingViewIndex < viewIdx) { //originating view is 'left'
                        if (m.queryIdx == originatingViewFeatureIndex) {
                            matched2DPointInNewView = m.trainIdx;
                        }
                    } else {                              //originating view is 'right'
                        if (m.trainIdx == originatingViewFeatureIndex) {
                            matched2DPointInNewView = m.queryIdx;
                        }
                    }
                    if (matched2DPointInNewView >= 0) {
                        //This point is matched in the new view
//                        cout << "found match to new 2d feature " << matched2DPointInNewView << endl;
                        const Features& newViewFeatures = mImageFeatures[viewIdx];
//                        cout << "leftViewIdx " << leftViewIdx << endl;
//                        cout << "rightViewIdx " << rightViewIdx << endl;
//                        cout << "newViewFeatures " << newViewFeatures.points.size() << endl;
//                        cout << "found 2d point " << newViewFeatures.points[matched2DPointInNewView] << endl;
                        match2D3D.points2D.push_back(newViewFeatures.points[matched2DPointInNewView]);
                        match2D3D.points3D.push_back(cloudPoint.p);
                        break;
                    }
                }
            }
        }

        matches[viewIdx] = match2D3D;
    }

    return matches;
}

void SfM::mergeNewPointCloud(const PointCloud& cloud) {
}

} /* namespace sfmtoylib */
